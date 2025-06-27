package dev.neuronic;

import dev.neuronic.net.NeuralNet;
import dev.neuronic.net.Layers;
import dev.neuronic.net.optimizers.SgdOptimizer;

/**
 * Test that the library runs without Vector API.
 * Run without vector: java -cp target/test-classes:target/classes dev.neuronic.TestNoVector
 * Run with vector: java --add-modules=jdk.incubator.vector -cp target/test-classes:target/classes dev.neuronic.TestNoVector
 */
public class TestNoVector {
    public static void main(String[] args) {
        System.out.println("=== Testing Neural Network without Vector API ===\n");
        
        try {
            // Build a simple XOR network
            NeuralNet net = NeuralNet.newBuilder()
                .input(2)
                .setDefaultOptimizer(new SgdOptimizer(0.5f))
                .layer(Layers.hiddenDenseRelu(3))
                .output(Layers.outputSoftmaxCrossEntropy(2));
            
            System.out.println("✓ Successfully created neural network");
            
            // Test data for XOR
            float[][] inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
            float[][] targets = {{1, 0}, {0, 1}, {0, 1}, {1, 0}};
            
            // Try a forward pass
            float[] output = net.predict(inputs[0]);
            System.out.println("✓ Successfully performed forward pass");
            System.out.printf("  Input: [%.1f, %.1f] -> Output: [%.3f, %.3f]\n", 
                inputs[0][0], inputs[0][1], output[0], output[1]);
            
            // Try training
            net.train(inputs[0], targets[0]);
            System.out.println("✓ Successfully performed training step");
            
            System.out.println("\n✓ SUCCESS - Library works without Vector API!");
            
        } catch (NoClassDefFoundError e) {
            System.err.println("\n✗ FAILED - NoClassDefFoundError:");
            System.err.println("  " + e.getMessage());
            System.err.println("\nThis means Vector API classes are being loaded even without --add-modules");
            e.printStackTrace();
            System.exit(1);
        } catch (Exception e) {
            System.err.println("\n✗ FAILED - Unexpected error:");
            e.printStackTrace();
            System.exit(1);
        }
    }
}