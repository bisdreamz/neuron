package dev.neuronic.benchmarks;

import dev.neuronic.net.math.ops.ElementwiseMultiply;

/**
 * Test to verify memory bandwidth is the bottleneck for large arrays.
 */
public class MemoryBandwidthTest {
    
    public static void main(String[] args) {
        System.out.println("=== Memory Bandwidth Analysis ===\n");
        
        // Test different sizes
        int[] sizes = {32, 128, 512, 2048, 8192, 32768};
        
        for (int size : sizes) {
            float[] a = new float[size];
            float[] b = new float[size];
            float[] output = new float[size];
            
            // Fill with data
            for (int i = 0; i < size; i++) {
                a[i] = i * 0.1f;
                b[i] = i * 0.2f;
            }
            
            // Warmup
            for (int i = 0; i < 10000; i++) {
                ElementwiseMultiply.computeScalar(a, b, output);
                ElementwiseMultiply.computeVectorized(a, b, output);
            }
            
            // Measure
            long scalarTime = timeOperation(() -> ElementwiseMultiply.computeScalar(a, b, output), 100000);
            long vectorTime = timeOperation(() -> ElementwiseMultiply.computeVectorized(a, b, output), 100000);
            
            // Calculate bandwidth (approximate)
            long bytesProcessed = size * 4L * 3; // 3 arrays * 4 bytes per float
            double scalarBandwidthGB = (bytesProcessed * 1e9 / scalarTime) / 1e9;
            double vectorBandwidthGB = (bytesProcessed * 1e9 / vectorTime) / 1e9;
            
            System.out.printf("Size %6d: Scalar=%4d ns (%.1f GB/s) | Vector=%4d ns (%.1f GB/s) | Speedup=%.2fx\n",
                size, scalarTime, scalarBandwidthGB, vectorTime, vectorBandwidthGB, 
                (double)scalarTime / vectorTime);
        }
        
        System.out.println("\nNote: Modern CPUs have ~25-50 GB/s memory bandwidth.");
        System.out.println("When approaching this limit, vectorization provides diminishing returns.");
    }
    
    private static long timeOperation(Runnable op, int iterations) {
        long start = System.nanoTime();
        for (int i = 0; i < iterations; i++) {
            op.run();
        }
        return (System.nanoTime() - start) / iterations;
    }
}