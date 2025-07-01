package dev.neuronic.net;

import dev.neuronic.net.optimizers.SgdOptimizer;
import org.junit.jupiter.api.Test;

import java.util.*;

/**
 * Test with extremely small learning rates to prevent gradient explosion.
 * TODO: REMOVE THIS FILE AFTER DEBUGGING
 */
public class MicroLearningRateGradientTest {
    
    @Test
    public void testMicroLearningRates() {
        System.out.println("=== MICRO LEARNING RATE GRADIENT EXPLOSION TEST ===\n");
        
        // Test increasingly smaller learning rates
        testLearningRate("LR_0_00001", 0.00001f);
        testLearningRate("LR_0_000001", 0.000001f);
        testLearningRate("LR_0_0000001", 0.0000001f);
        testLearningRate("LR_0_00000001", 0.00000001f);
    }
    
    @Test
    public void testInitializationScales() {
        System.out.println("=== WEIGHT INITIALIZATION SCALE TEST ===\n");
        
        // Standard initialization might be too large
        // Test with different initialization by training very little
        testMinimalTraining("MINIMAL_1_STEP", 1);
        testMinimalTraining("MINIMAL_10_STEPS", 10);
        testMinimalTraining("MINIMAL_100_STEPS", 100);
    }
    
    private void testLearningRate(String name, float lr) {
        System.out.println("--- " + name + " (LR=" + lr + ") ---");
        
        // Ultra-simple network to isolate the issue
        NeuralNet net = NeuralNet.newBuilder()
            .input(2)
            .setDefaultOptimizer(new SgdOptimizer(lr))
            .layer(Layers.hiddenDenseRelu(4)) // Very small hidden layer
            .output(Layers.outputLinearRegression(1));
        
        Random rand = new Random(42);
        boolean exploded = false;
        
        // Monitor gradients very closely
        for (int step = 0; step < 100; step++) {
            float[] input = {rand.nextFloat(), rand.nextFloat()};
            float target = (input[0] > input[1]) ? 1.0f : 0.0f;
            
            // Get prediction before training
            float predBefore = net.predict(input)[0];
            
            // Train single step
            net.trainBatch(new float[][]{input}, new float[][]{new float[]{target}});
            
            // Get prediction after training
            float predAfter = net.predict(input)[0];
            
            // Check for explosion
            if (Float.isNaN(predAfter) || Float.isInfinite(predAfter) || Math.abs(predAfter) > 1000) {
                System.out.printf("  EXPLODED at step %d: %.6f → %.6f\n", step, predBefore, predAfter);
                exploded = true;
                break;
            }
            
            // Log first few steps
            if (step < 5) {
                System.out.printf("  Step %d: %.6f → %.6f (change: %.6f)\n", 
                    step, predBefore, predAfter, predAfter - predBefore);
            }
        }
        
        if (!exploded) {
            // Test final predictions
            Set<String> uniquePreds = new HashSet<>();
            for (int i = 0; i < 20; i++) {
                float[] input = {i < 10 ? 0.8f : 0.2f, i < 10 ? 0.2f : 0.8f};
                float pred = net.predict(input)[0];
                uniquePreds.add(String.format("%.4f", pred));
            }
            
            System.out.printf("  Final unique predictions: %d\n", uniquePreds.size());
            System.out.printf("  Status: %s\n", uniquePreds.size() > 3 ? "✓ STABLE" : "⚠️ COLLAPSED");
        }
        
        System.out.println();
    }
    
    private void testMinimalTraining(String name, int maxSteps) {
        System.out.println("--- " + name + " ---");
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(2)
            .setDefaultOptimizer(new SgdOptimizer(0.01f)) // Normal LR but very few steps
            .layer(Layers.hiddenDenseRelu(8))
            .output(Layers.outputLinearRegression(1));
        
        Random rand = new Random(42);
        
        System.out.println("  Initial predictions:");
        for (int i = 0; i < 5; i++) {
            float[] input = {rand.nextFloat(), rand.nextFloat()};
            float pred = net.predict(input)[0];
            System.out.printf("    [%.3f, %.3f] → %.6f\n", input[0], input[1], pred);
        }
        
        // Train for very few steps
        for (int step = 0; step < maxSteps; step++) {
            float[] input = {rand.nextFloat(), rand.nextFloat()};
            float target = (input[0] > input[1]) ? 1.0f : 0.0f;
            net.trainBatch(new float[][]{input}, new float[][]{new float[]{target}});
        }
        
        System.out.printf("  After %d training steps:\n", maxSteps);
        rand = new Random(42); // Reset for same test inputs
        boolean exploded = false;
        
        for (int i = 0; i < 5; i++) {
            float[] input = {rand.nextFloat(), rand.nextFloat()};
            float pred = net.predict(input)[0];
            
            if (Float.isNaN(pred) || Float.isInfinite(pred) || Math.abs(pred) > 1000) {
                System.out.printf("    [%.3f, %.3f] → EXPLODED (%.6f)\n", input[0], input[1], pred);
                exploded = true;
            } else {
                System.out.printf("    [%.3f, %.3f] → %.6f\n", input[0], input[1], pred);
            }
        }
        
        System.out.printf("  Result: %s\n\n", exploded ? "⚠️ EXPLODED" : "✓ STABLE");
    }
    
    @Test
    public void testGradientClippingEffectiveness() {
        System.out.println("=== GRADIENT CLIPPING EFFECTIVENESS TEST ===\n");
        
        // Test if gradient clipping is actually preventing explosion
        NeuralNet net = NeuralNet.newBuilder()
            .input(2)
            .setDefaultOptimizer(new SgdOptimizer(0.1f)) // High LR to trigger explosion
            .layer(Layers.hiddenDenseRelu(16))
            .output(Layers.outputLinearRegression(1));
        
        Random rand = new Random(42);
        
        System.out.println("Training with high LR (0.1) and gradient clipping...");
        System.out.println("If clipping works, predictions should stay reasonable.");
        System.out.println("If clipping fails, predictions will explode despite warnings.");
        
        for (int step = 0; step < 50; step++) {
            float[] input = {rand.nextFloat() * 10, rand.nextFloat() * 10}; // Large inputs
            float target = (input[0] > input[1]) ? 10.0f : -10.0f; // Large targets
            
            float predBefore = net.predict(input)[0];
            net.trainBatch(new float[][]{input}, new float[][]{new float[]{target}});
            float predAfter = net.predict(input)[0];
            
            if (step < 10 || step % 10 == 0) {
                System.out.printf("  Step %d: %.6f → %.6f (change: %.6f)\n", 
                    step, predBefore, predAfter, predAfter - predBefore);
            }
            
            // Stop if exploded
            if (Float.isNaN(predAfter) || Float.isInfinite(predAfter) || Math.abs(predAfter) > 1e6) {
                System.out.printf("  EXPLODED at step %d despite gradient clipping!\n", step);
                break;
            }
        }
        
        System.out.println();
    }
}