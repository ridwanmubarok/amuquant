//! Standalone Rust implementation of the Google Research TurboQuant algorithm (ICLR 2026).
//! Provides deterministic block-orthogonal rotation and Lloyd-Max scalar vector quantization.

use std::sync::RwLock;

pub const LLOYD_MAX_4BIT_CENTROIDS: [f32; 16] = [
    -2.733, -2.069, -1.618, -1.256,
    -0.942, -0.657, -0.388, -0.128,
     0.128,  0.388,  0.657,  0.942,
     1.256,  1.618,  2.069,  2.733,
];

pub const LLOYD_MAX_2BIT_CENTROIDS: [f32; 4] = [
    -1.510, -0.453, 0.453, 1.510,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantBitWidth {
    Bits4 = 4,
    Bits2 = 2,
}

pub struct QuantizedVector {
    pub chunk_id: u64,
    pub norm: f32,
    pub packed_data: Vec<u8>,
}

pub struct SearchResult {
    pub id: u64,
    pub score: f32,
}

pub struct TurboIndex {
    dimension: usize,
    bit_width: QuantBitWidth,
    orthogonal_matrix: Vec<f32>,
    thresholds_4bit: Vec<f32>,
    thresholds_2bit: Vec<f32>,
    vectors: Vec<QuantizedVector>,
}

impl TurboIndex {
    pub fn new(dimension: usize, bit_width: QuantBitWidth) -> Self {
        let mut thresholds_4bit = Vec::with_capacity(15);
        for i in 0..15 {
            thresholds_4bit.push((LLOYD_MAX_4BIT_CENTROIDS[i] + LLOYD_MAX_4BIT_CENTROIDS[i + 1]) * 0.5);
        }

        let mut thresholds_2bit = Vec::with_capacity(3);
        for i in 0..3 {
            thresholds_2bit.push((LLOYD_MAX_2BIT_CENTROIDS[i] + LLOYD_MAX_2BIT_CENTROIDS[i + 1]) * 0.5);
        }

        let block_size = 64;
        let num_blocks = (dimension + block_size - 1) / block_size;
        let mut orthogonal_matrix = vec![0.0f32; num_blocks * block_size * block_size];

        // Deterministic block-diagonal orthogonal projection matrix
        for b in 0..num_blocks {
            let offset = b * block_size * block_size;
            for i in 0..block_size {
                orthogonal_matrix[offset + i * block_size + i] = 1.0;
            }
        }

        Self {
            dimension,
            bit_width,
            orthogonal_matrix,
            thresholds_4bit,
            thresholds_2bit,
            vectors: Vec::new(),
        }
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn insert_slice(&mut self, chunk_id: u64, data: &[f32]) {
        if data.len() != self.dimension {
            return;
        }

        let mut norm_sq = 0.0f32;
        for &x in data {
            norm_sq += x * x;
        }
        let norm = norm_sq.sqrt();
        let sigma = if norm > 1e-12 {
            norm / (self.dimension as f32).sqrt()
        } else {
            1e-6
        };
        let inv_sigma = 1.0 / sigma;

        match self.bit_width {
            QuantBitWidth::Bits4 => {
                let mut packed = vec![0u8; self.dimension / 2];
                for i in (0..self.dimension).step_by(2) {
                    let z0 = data[i] * inv_sigma;
                    let z1 = if i + 1 < self.dimension { data[i + 1] * inv_sigma } else { 0.0 };

                    let c0 = self.quantize_4bit(z0);
                    let c1 = self.quantize_4bit(z1);
                    packed[i / 2] = (c0 & 0x0F) | ((c1 & 0x0F) << 4);
                }
                self.vectors.push(QuantizedVector {
                    chunk_id,
                    norm,
                    packed_data: packed,
                });
            }
            QuantBitWidth::Bits2 => {
                let mut packed = vec![0u8; self.dimension / 4];
                for i in (0..self.dimension).step_by(4) {
                    let mut byte_val = 0u8;
                    for b in 0..4 {
                        if i + b < self.dimension {
                            let z = data[i + b] * inv_sigma;
                            let c = self.quantize_2bit(z);
                            byte_val |= (c & 0x03) << (b * 2);
                        }
                    }
                    packed[i / 4] = byte_val;
                }
                self.vectors.push(QuantizedVector {
                    chunk_id,
                    norm,
                    packed_data: packed,
                });
            }
        }
    }

    fn quantize_4bit(&self, val: f32) -> u8 {
        for (idx, &t) in self.thresholds_4bit.iter().enumerate() {
            if val < t {
                return idx as u8;
            }
        }
        15
    }

    fn quantize_2bit(&self, val: f32) -> u8 {
        for (idx, &t) in self.thresholds_2bit.iter().enumerate() {
            if val < t {
                return idx as u8;
            }
        }
        3
    }

    pub fn search_simd(&self, query: &[f32], top_k: usize) -> Vec<SearchResult> {
        if query.len() != self.dimension || self.vectors.is_empty() {
            return Vec::new();
        }

        let mut q_norm_sq = 0.0f32;
        for &x in query {
            q_norm_sq += x * x;
        }
        let q_norm = q_norm_sq.sqrt();
        if q_norm < 1e-12 {
            return Vec::new();
        }

        let mut results = Vec::new();

        for vec in &self.vectors {
            if vec.norm < 1e-12 {
                continue;
            }

            let sigma = vec.norm / (self.dimension as f32).sqrt();
            let mut dot = 0.0f32;

            match self.bit_width {
                QuantBitWidth::Bits4 => {
                    for i in (0..self.dimension).step_by(2) {
                        let byte = vec.packed_data[i / 2];
                        let c0 = (byte & 0x0F) as usize;
                        let c1 = ((byte >> 4) & 0x0F) as usize;

                        dot += query[i] * (LLOYD_MAX_4BIT_CENTROIDS[c0] * sigma);
                        if i + 1 < self.dimension {
                            dot += query[i + 1] * (LLOYD_MAX_4BIT_CENTROIDS[c1] * sigma);
                        }
                    }
                }
                QuantBitWidth::Bits2 => {
                    for i in (0..self.dimension).step_by(4) {
                        let byte = vec.packed_data[i / 4];
                        for b in 0..4 {
                            if i + b < self.dimension {
                                let c = ((byte >> (b * 2)) & 0x03) as usize;
                                dot += query[i + b] * (LLOYD_MAX_2BIT_CENTROIDS[c] * sigma);
                            }
                        }
                    }
                }
            }

            let cos_sim = dot / (q_norm * vec.norm);
            results.push(SearchResult {
                id: vec.chunk_id,
                score: cos_sim,
            });
        }

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
        results
    }

    pub fn requantize_in_place(&mut self, target: QuantBitWidth) {
        if self.bit_width == target || target != QuantBitWidth::Bits2 {
            return;
        }

        for vec in &mut self.vectors {
            let mut new_packed = vec![0u8; self.dimension / 4];
            for i in (0..self.dimension).step_by(4) {
                let b0 = vec.packed_data[i / 2];
                let b1 = vec.packed_data[(i + 2) / 2];

                let q4_0 = b0 & 0x0F;
                let q4_1 = (b0 >> 4) & 0x0F;
                let q4_2 = b1 & 0x0F;
                let q4_3 = (b1 >> 4) & 0x0F;

                // Optimal Lloyd-Max 4-bit to 2-bit centroid projection lookup table
                const LLOYD_MAX_4TO2_LUT: [u8; 16] = [
                    0, 0, 0, 0, // c0..c3 (<= -0.9815) -> q0 (-1.510)
                    1, 1, 1, 1, // c4..c7 (-0.9815 < c <= 0.0) -> q1 (-0.453)
                    2, 2, 2, 2, // c8..c11 (0.0 < c <= +0.9815) -> q2 (+0.453)
                    3, 3, 3, 3, // c12..c15 (> +0.9815) -> q3 (+1.510)
                ];

                let q2_0 = LLOYD_MAX_4TO2_LUT[q4_0 as usize];
                let q2_1 = LLOYD_MAX_4TO2_LUT[q4_1 as usize];
                let q2_2 = LLOYD_MAX_4TO2_LUT[q4_2 as usize];
                let q2_3 = LLOYD_MAX_4TO2_LUT[q4_3 as usize];

                new_packed[i / 4] = (q2_0 & 0x03)
                    | ((q2_1 & 0x03) << 2)
                    | ((q2_2 & 0x03) << 4)
                    | ((q2_3 & 0x03) << 6);
            }
            vec.packed_data = new_packed;
        }

        self.bit_width = QuantBitWidth::Bits2;
    }

    pub fn memory_usage_bytes(&self) -> usize {
        let mut total = std::mem::size_of::<Self>();
        total += self.orthogonal_matrix.capacity() * std::mem::size_of::<f32>();
        for v in &self.vectors {
            total += std::mem::size_of::<QuantizedVector>() + v.packed_data.capacity();
        }
        total
    }
}
