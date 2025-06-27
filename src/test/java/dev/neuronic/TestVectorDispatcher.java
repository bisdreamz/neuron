package dev.neuronic;

import dev.neuronic.net.math.ops.ElementwiseAdd;
import dev.neuronic.net.math.ops.DotProduct;

/**
 * Test the VectorDispatcher pattern.
 * Run without vector: java -cp target/test-classes:target/classes dev.neuronic.TestVectorDispatcher
 * Run with vector: java --add-modules=jdk.incubator.vector -cp target/test-classes:target/classes dev.neuronic.TestVectorDispatcher
 */
public class TestVectorDispatcher {
    public static void main(String[] args) {
        System.out.println("=== Testing VectorDispatcher Pattern ===\n");
        
        // Test data
        float[] a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
        float[] b = {8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
        float[] result = new float[8];
        
        // Test ElementwiseAdd
        ElementwiseAdd.compute(a, b, result);
        System.out.print("ElementwiseAdd result: [");
        for (int i = 0; i < result.length; i++) {
            System.out.print(result[i]);
            if (i < result.length - 1) System.out.print(", ");
        }
        System.out.println("]");
        System.out.println("Expected: [9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0]");
        
        // Test DotProduct
        float dotResult = DotProduct.compute(a, b);
        System.out.println("\nDotProduct result: " + dotResult);
        System.out.println("Expected: 120.0");
        
        // Verify results
        boolean addCorrect = true;
        for (float v : result) {
            if (Math.abs(v - 9.0f) > 1e-6) {
                addCorrect = false;
                break;
            }
        }
        boolean dotCorrect = Math.abs(dotResult - 120.0f) < 1e-6;
        
        if (addCorrect && dotCorrect) {
            System.out.println("\n✓ SUCCESS - VectorDispatcher pattern works correctly!");
            System.out.println("  No ClassNotFoundException when Vector API is not available");
        } else {
            System.out.println("\n✗ FAILED - Incorrect results");
        }
    }
}