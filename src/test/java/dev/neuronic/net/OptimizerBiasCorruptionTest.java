package dev.neuronic.net;

import dev.neuronic.net.optimizers.AdamWOptimizer;
import dev.neuronic.net.optimizers.AdamOptimizer;
import dev.neuronic.net.optimizers.SgdOptimizer;
import org.junit.jupiter.api.Test;

import java.util.*;

/**
 * Test Adam/AdamW bias correction numerical stability and state corruption.
 * TODO: REMOVE THIS FILE AFTER DEBUGGING
 */
public class OptimizerBiasCorruptionTest {
    
    @Test
    public void testAdamBiasCorrectionUnderflow() {
        System.out.println("=== ADAM BIAS CORRECTION UNDERFLOW TEST ===\n");
        
        // Test bias correction formula: 1 - beta^t
        float beta1 = 0.9f;
        float beta2 = 0.999f;
        
        System.out.println("Testing bias correction stability:");
        System.out.println("Beta1 = " + beta1 + ", Beta2 = " + beta2);
        System.out.println();
        
        // Test for various time steps
        int[] steps = {1, 10, 100, 1000, 10000, 100000, 1000000};
        
        for (int t : steps) {
            double bias1 = 1.0 - Math.pow(beta1, t);
            double bias2 = 1.0 - Math.pow(beta2, t);
            
            // Check for underflow
            boolean underflow1 = bias1 == 0.0 || Double.isInfinite(1.0 / bias1);
            boolean underflow2 = bias2 == 0.0 || Double.isInfinite(1.0 / bias2);
            
            System.out.printf("Step %d:\n", t);
            System.out.printf("  1 - beta1^t = %.15e %s\n", bias1, underflow1 ? "⚠️ UNDERFLOW!" : "");
            System.out.printf("  1 - beta2^t = %.15e %s\n", bias2, underflow2 ? "⚠️ UNDERFLOW!" : "");
            System.out.printf("  1/(1-beta1^t) = %.6f\n", 1.0 / bias1);
            System.out.printf("  1/(1-beta2^t) = %.6f\n", 1.0 / bias2);
            System.out.println();
        }
    }
    
    @Test
    public void testOptimizerStateMonitoring() {
        System.out.println("=== OPTIMIZER STATE MONITORING TEST ===\n");
        
        // Create simple network with AdamW
        AdamWOptimizer optimizer = new AdamWOptimizer(0.001f, 0.0f); // No weight decay
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(2)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.hiddenDenseRelu(4))
            .output(Layers.outputLinearRegression(1));
        
        Random rand = new Random(42);
        
        // Monitor state evolution
        for (int checkpoint = 0; checkpoint <= 1000; checkpoint += 100) {
            // Train 100 steps
            for (int i = 0; i < 100 && checkpoint > 0; i++) {
                float[] input = {rand.nextFloat(), rand.nextFloat()};
                float target = (input[0] > input[1]) ? 1.0f : 0.0f;
                
                try {
                    net.trainBatch(new float[][]{input}, new float[][]{new float[]{target}});
                } catch (Exception e) {
                    System.out.printf("ERROR at step %d: %s\n", checkpoint + i, e.getMessage());
                    return;
                }
            }
            
            // Check predictions for NaN/Inf
            float[] testInput = {0.5f, 0.5f};
            float pred = net.predict(testInput)[0];
            
            System.out.printf("Step %d: prediction = %.6f", checkpoint, pred);
            
            if (Float.isNaN(pred)) {
                System.out.println(" ⚠️ NaN DETECTED!");
                break;
            } else if (Float.isInfinite(pred)) {
                System.out.println(" ⚠️ INFINITE DETECTED!");
                break;
            } else if (Math.abs(pred) > 1e6) {
                System.out.println(" ⚠️ EXPLOSION DETECTED!");
                break;
            }
            
            // Test diversity
            Set<String> uniquePreds = new HashSet<>();
            for (int i = 0; i < 20; i++) {
                float[] input = {i < 10 ? 0.8f : 0.2f, i < 10 ? 0.2f : 0.8f};
                float p = net.predict(input)[0];
                uniquePreds.add(String.format("%.6f", p));
            }
            
            System.out.printf(" (unique=%d)", uniquePreds.size());
            if (uniquePreds.size() <= 2) {
                System.out.print(" ⚠️ COLLAPSED");
            }
            System.out.println();
        }
    }
    
    @Test
    public void testOptimizerStateSharingBug() {
        System.out.println("=== OPTIMIZER STATE SHARING BUG TEST ===\n");
        
        // Test if optimizer state is incorrectly shared between layers
        AdamWOptimizer optimizer = new AdamWOptimizer(0.01f, 0.0f);
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(2)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.hiddenDenseRelu(4))
            .layer(Layers.hiddenDenseRelu(4)) // Same size - might share state?
            .output(Layers.outputLinearRegression(1));
        
        Random rand = new Random(42);
        
        System.out.println("Training network with multiple layers of same size...");
        System.out.println("If state is incorrectly shared, collapse will be faster.\n");
        
        for (int step = 0; step <= 500; step += 50) {
            // Train 50 steps
            for (int i = 0; i < 50 && step > 0; i++) {
                float[] input = {rand.nextFloat(), rand.nextFloat()};
                float target = (input[0] > input[1]) ? 1.0f : 0.0f;
                net.trainBatch(new float[][]{input}, new float[][]{new float[]{target}});
            }
            
            // Test diversity
            Set<String> uniquePreds = new HashSet<>();
            for (int i = 0; i < 20; i++) {
                float[] input = {i < 10 ? 0.9f : 0.1f, i < 10 ? 0.1f : 0.9f};
                float pred = net.predict(input)[0];
                uniquePreds.add(String.format("%.4f", pred));
            }
            
            System.out.printf("Step %d: %d unique predictions %s\n", 
                step, uniquePreds.size(), 
                uniquePreds.size() <= 2 ? "⚠️ COLLAPSED" : "✓");
            
            if (uniquePreds.size() <= 2) break;
        }
    }
    
    @Test
    public void testTimeStepCorruption() {
        System.out.println("=== TIME STEP CORRUPTION TEST ===\n");
        
        // Test if time step counter is getting corrupted
        System.out.println("Creating network and monitoring time step behavior...\n");
        
        // Simple test: train for many steps and see when things break
        NeuralNet net = NeuralNet.newBuilder()
            .input(1)
            .setDefaultOptimizer(new AdamWOptimizer(0.001f, 0.0f))
            .output(Layers.outputLinearRegression(1)); // Simplest possible
        
        Random rand = new Random(42);
        
        // Train and monitor
        int[] checkpoints = {1, 10, 100, 1000, 10000, 50000};
        int trained = 0;
        
        for (int checkpoint : checkpoints) {
            // Train to checkpoint
            while (trained < checkpoint) {
                float[] input = {rand.nextFloat()};
                float target = input[0] * 2; // Simple linear relationship
                
                net.trainBatch(new float[][]{input}, new float[][]{new float[]{target}});
                trained++;
            }
            
            // Test prediction
            float[] testInput = {0.5f};
            float pred = net.predict(testInput)[0];
            float expected = 1.0f; // 0.5 * 2
            float error = Math.abs(pred - expected);
            
            System.out.printf("Step %d: pred=%.6f, expected=%.6f, error=%.6f", 
                checkpoint, pred, expected, error);
            
            if (Float.isNaN(pred) || Float.isInfinite(pred)) {
                System.out.println(" ⚠️ NaN/Inf DETECTED!");
                break;
            } else if (error > 10.0f) {
                System.out.println(" ⚠️ LARGE ERROR!");
            } else {
                System.out.println(" ✓");
            }
        }
    }
}