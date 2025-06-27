package dev.neuronic;

import dev.neuronic.net.math.ops.DotProduct;

/**
 * Test that DotProduct works without Vector API.
 * Run: java -cp target/test-classes:target/classes dev.neuronic.TestVectorOptional
 */
public class TestVectorOptional {
    public static void main(String[] args) {
        System.out.println("=== Testing DotProduct without Vector API ===\n");
        
        float[] a = {1.0f, 2.0f, 3.0f, 4.0f};
        float[] b = {5.0f, 6.0f, 7.0f, 8.0f};
        
        float result = DotProduct.compute(a, b);
        System.out.println("DotProduct result: " + result);
        System.out.println("Expected: 70.0 (1*5 + 2*6 + 3*7 + 4*8)");
        
        if (Math.abs(result - 70.0f) < 1e-6) {
            System.out.println("\n✓ Success - DotProduct works without Vector API!");
        } else {
            System.out.println("\n✗ Failed - incorrect result");
        }
    }
}