package dev.neuronic;

import dev.neuronic.net.math.VectorizationNew;
import dev.neuronic.net.math.VectorProvider;

/**
 * Test the provider pattern for optional Vector API.
 * Run without vector: java -cp target/test-classes:target/classes dev.neuronic.TestProviderPattern
 * Run with vector: java --add-modules=jdk.incubator.vector -cp target/test-classes:target/classes dev.neuronic.TestProviderPattern
 */
public class TestProviderPattern {
    public static void main(String[] args) {
        System.out.println("=== Testing Provider Pattern for Optional Vector API ===\n");
        
        // Test initialization
        System.out.println("Vector API available: " + VectorizationNew.isAvailable());
        System.out.println("Vector length: " + VectorizationNew.getVectorLength());
        
        // Get provider
        VectorProvider provider = VectorizationNew.getProvider();
        System.out.println("Provider class: " + provider.getClass().getSimpleName());
        
        // Test operations
        float[] a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
        float[] b = {8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
        float[] result = new float[8];
        
        // Test dot product
        float dotResult = provider.dotProduct(a, b);
        System.out.println("\nDot product result: " + dotResult);
        System.out.println("Expected: 120.0 (1*8 + 2*7 + ... + 8*1)");
        
        // Test element-wise add
        provider.elementwiseAdd(a, b, result);
        System.out.print("\nElement-wise add result: [");
        for (int i = 0; i < result.length; i++) {
            System.out.print(result[i]);
            if (i < result.length - 1) System.out.print(", ");
        }
        System.out.println("]");
        System.out.println("Expected: [9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0]");
        
        // Verify results
        boolean dotCorrect = Math.abs(dotResult - 120.0f) < 1e-6;
        boolean addCorrect = true;
        for (float v : result) {
            if (Math.abs(v - 9.0f) > 1e-6) {
                addCorrect = false;
                break;
            }
        }
        
        if (dotCorrect && addCorrect) {
            System.out.println("\n✓ SUCCESS - Provider pattern works correctly!");
        } else {
            System.out.println("\n✗ FAILED - Incorrect results");
        }
    }
}