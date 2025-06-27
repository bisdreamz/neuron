package dev.neuronic.benchmarks;

import dev.neuronic.net.math.ops.DotProduct;
import dev.neuronic.net.math.ops.ElementwiseMultiply;
import dev.neuronic.net.math.ops.ElementwiseAdd;
import dev.neuronic.net.math.Vectorization;

import java.util.concurrent.ThreadLocalRandom;

/**
 * Direct comparison of scalar vs vectorized implementations.
 * Focused on realistic ad prediction model sizes.
 */
public class ScalarVsVectorizedBenchmark {
    
    public static void main(String[] args) {
        System.out.println("=== Scalar vs Vectorized Performance Comparison ===");
        System.out.println("Vector API available: " + Vectorization.isAvailable());
        System.out.println("Vector length: " + Vectorization.getVectorLength());
        System.out.println();
        
        // Test sizes relevant to ad prediction and small models
        int[] sizes = {8, 16, 32, 64, 128, 256, 512, 1024};
        
        for (int size : sizes) {
            System.out.println("=== Size: " + size + " ===");
            benchmarkDotProduct(size);
            benchmarkElementwiseMultiply(size);
            benchmarkElementwiseAdd(size);
            System.out.println();
        }
    }
    
    private static void benchmarkDotProduct(int size) {
        float[] a = createRandomArray(size);
        float[] b = createRandomArray(size);
        
        // Warmup both implementations
        for (int i = 0; i < 50000; i++) {
            DotProduct.computeScalar(a, b);
            if (Vectorization.isAvailable()) {
                DotProduct.computeVectorized(a, b);
            }
        }
        
        // Benchmark scalar
        long scalarTime = timeOperation(() -> DotProduct.computeScalar(a, b), 200000);
        
        // Benchmark vectorized  
        long vectorizedTime = Long.MAX_VALUE;
        if (Vectorization.isAvailable()) {
            vectorizedTime = timeOperation(() -> DotProduct.computeVectorized(a, b), 200000);
        }
        
        System.out.printf("DotProduct:     Scalar=%4d ns | Vectorized=%4d ns", scalarTime, vectorizedTime);
        if (Vectorization.isAvailable() && vectorizedTime > 0) {
            System.out.printf(" | Speedup=%.2fx", (double)scalarTime / vectorizedTime);
        }
        System.out.println();
    }
    
    private static void benchmarkElementwiseMultiply(int size) {
        float[] a = createRandomArray(size);
        float[] b = createRandomArray(size);
        float[] output = new float[size];
        
        // Warmup
        for (int i = 0; i < 50000; i++) {
            ElementwiseMultiply.computeScalar(a, b, output);
            if (Vectorization.isAvailable()) {
                ElementwiseMultiply.computeVectorized(a, b, output);
            }
        }
        
        // Benchmark scalar
        long scalarTime = timeOperation(() -> ElementwiseMultiply.computeScalar(a, b, output), 200000);
        
        // Benchmark vectorized
        long vectorizedTime = Long.MAX_VALUE;
        if (Vectorization.isAvailable()) {
            vectorizedTime = timeOperation(() -> ElementwiseMultiply.computeVectorized(a, b, output), 200000);
        }
        
        System.out.printf("ElementMultiply: Scalar=%4d ns | Vectorized=%4d ns", scalarTime, vectorizedTime);
        if (Vectorization.isAvailable() && vectorizedTime > 0) {
            System.out.printf(" | Speedup=%.2fx", (double)scalarTime / vectorizedTime);
        }
        System.out.println();
    }
    
    private static void benchmarkElementwiseAdd(int size) {
        float[] a = createRandomArray(size);
        float[] b = createRandomArray(size);
        float[] output = new float[size];
        
        // Warmup
        for (int i = 0; i < 50000; i++) {
            ElementwiseAdd.computeScalar(a, b, output);
            if (Vectorization.isAvailable()) {
                ElementwiseAdd.computeVectorized(a, b, output);
            }
        }
        
        // Benchmark scalar
        long scalarTime = timeOperation(() -> ElementwiseAdd.computeScalar(a, b, output), 200000);
        
        // Benchmark vectorized
        long vectorizedTime = Long.MAX_VALUE;
        if (Vectorization.isAvailable()) {
            vectorizedTime = timeOperation(() -> ElementwiseAdd.computeVectorized(a, b, output), 200000);
        }
        
        System.out.printf("ElementAdd:     Scalar=%4d ns | Vectorized=%4d ns", scalarTime, vectorizedTime);
        if (Vectorization.isAvailable() && vectorizedTime > 0) {
            System.out.printf(" | Speedup=%.2fx", (double)scalarTime / vectorizedTime);
        }
        System.out.println();
    }
    
    private static float[] createRandomArray(int size) {
        ThreadLocalRandom random = ThreadLocalRandom.current();
        float[] array = new float[size];
        
        for (int i = 0; i < size; i++) {
            array[i] = (float) (random.nextGaussian() * 0.1);
        }
        return array;
    }
    
    private static long timeOperation(Runnable operation, int iterations) {
        long startTime = System.nanoTime();
        
        for (int i = 0; i < iterations; i++) {
            operation.run();
        }
        
        long endTime = System.nanoTime();
        return (endTime - startTime) / iterations;
    }
}