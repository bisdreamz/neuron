package dev.neuronic.net;

import dev.neuronic.net.optimizers.SgdOptimizer;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import org.junit.jupiter.api.Test;

import java.util.*;

/**
 * Test if optimizer state accumulation is causing the collapse over time.
 * TODO: REMOVE THIS FILE AFTER DEBUGGING
 */
public class OptimizerStateCorruptionTest {
    
    @Test
    public void testOptimizerStateAccumulation() {
        System.out.println("=== OPTIMIZER STATE ACCUMULATION TEST ===\n");
        
        // Test if different optimizers accumulate corruption differently
        testOptimizerSteps("SGD", new SgdOptimizer(0.01f));
        testOptimizerSteps("ADAMW", new AdamWOptimizer(0.01f, 0.001f));
    }
    
    @Test
    public void testWeightQuantization() {
        System.out.println("=== WEIGHT QUANTIZATION TEST ===\n");
        
        // Create a network and monitor actual weight values over time
        NeuralNet net = NeuralNet.newBuilder()
            .input(2)
            .setDefaultOptimizer(new SgdOptimizer(0.001f)) // Very small LR
            .layer(Layers.hiddenDenseRelu(4))
            .output(Layers.outputLinearRegression(1));
        
        Random rand = new Random(42);
        
        // Sample a few weight values to monitor
        System.out.println("Monitoring weight evolution...");
        
        for (int step = 0; step <= 2000; step += 100) {
            if (step > 0) {
                // Train 100 steps
                for (int i = 0; i < 100; i++) {
                    float[] input = {rand.nextFloat(), rand.nextFloat()};
                    float target = (input[0] > input[1]) ? 1.0f : 0.0f;
                    net.trainBatch(new float[][]{input}, new float[][]{new float[]{target}});
                }
            }
            
            // Test predictions and count unique values
            Set<String> uniquePreds = new HashSet<>();
            float sumPred = 0;
            
            for (int i = 0; i < 100; i++) {
                float[] input = {i < 50 ? 0.8f : 0.2f, i < 50 ? 0.2f : 0.8f};
                float pred = net.predict(input)[0];
                uniquePreds.add(String.format("%.8f", pred)); // High precision
                sumPred += pred;
            }
            
            float avgPred = sumPred / 100;
            
            System.out.printf("Step %d: Unique=%-3d, Avg=%.6f", 
                step, uniquePreds.size(), avgPred);
            
            if (uniquePreds.size() <= 5) {
                System.out.print(" ⚠️ COLLAPSED");
                
                // Show the few unique values
                System.out.print(" Values: ");
                int count = 0;
                for (String pred : uniquePreds) {
                    if (count++ >= 3) break;
                    System.out.print(pred + " ");
                }
            }
            
            System.out.println();
            
            // Stop if collapsed
            if (uniquePreds.size() <= 2) {
                System.out.println("COLLAPSE DETECTED - stopping test");
                break;
            }
        }
    }
    
    @Test
    public void testFreshOptimizerEveryN() {
        System.out.println("=== FRESH OPTIMIZER RESET TEST ===\n");
        
        // Test if creating a fresh optimizer every N steps prevents collapse
        testWithOptimizerReset("RESET_EVERY_100", 100);
        testWithOptimizerReset("RESET_EVERY_500", 500);
        testWithOptimizerReset("NO_RESET", Integer.MAX_VALUE);
    }
    
    private void testOptimizerSteps(String optimizerName, dev.neuronic.net.optimizers.Optimizer optimizer) {
        System.out.println("--- " + optimizerName + " ---");
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(2)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.hiddenDenseRelu(8))
            .output(Layers.outputLinearRegression(1));
        
        Random rand = new Random(42);
        
        // Track when collapse happens
        for (int checkpoint = 100; checkpoint <= 2000; checkpoint += 100) {
            // Train to this checkpoint
            while (getTrainingStepCount(net) < checkpoint) {
                float[] input = {rand.nextFloat(), rand.nextFloat()};
                float target = (input[0] > input[1]) ? 1.0f : 0.0f;
                net.trainBatch(new float[][]{input}, new float[][]{new float[]{target}});
            }
            
            // Test diversity
            Set<String> uniquePreds = new HashSet<>();
            for (int i = 0; i < 50; i++) {
                float[] input = {i < 25 ? 0.9f : 0.1f, i < 25 ? 0.1f : 0.9f};
                float pred = net.predict(input)[0];
                uniquePreds.add(String.format("%.6f", pred));
            }
            
            System.out.printf("  Step %d: %d unique predictions", checkpoint, uniquePreds.size());
            
            if (uniquePreds.size() <= 3) {
                System.out.println(" ⚠️ COLLAPSED");
                break;
            } else {
                System.out.println(" ✓");
            }
        }
        
        System.out.println();
    }
    
    private void testWithOptimizerReset(String testName, int resetInterval) {
        System.out.println("--- " + testName + " ---");
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(2)
            .setDefaultOptimizer(new SgdOptimizer(0.01f))
            .layer(Layers.hiddenDenseRelu(8))
            .output(Layers.outputLinearRegression(1));
        
        Random rand = new Random(42);
        
        for (int step = 0; step < 1000; step++) {
            // Reset optimizer every N steps
            if (step > 0 && step % resetInterval == 0) {
                // Create fresh optimizer (simulating reset of internal state)
                net = NeuralNet.newBuilder()
                    .input(2)
                    .setDefaultOptimizer(new SgdOptimizer(0.01f))
                    .layer(Layers.hiddenDenseRelu(8))
                    .output(Layers.outputLinearRegression(1));
                
                System.out.printf("  Reset optimizer at step %d\n", step);
            }
            
            float[] input = {rand.nextFloat(), rand.nextFloat()};
            float target = (input[0] > input[1]) ? 1.0f : 0.0f;
            net.trainBatch(new float[][]{input}, new float[][]{new float[]{target}});
        }
        
        // Test final diversity
        Set<String> uniquePreds = new HashSet<>();
        for (int i = 0; i < 50; i++) {
            float[] input = {i < 25 ? 0.9f : 0.1f, i < 25 ? 0.1f : 0.9f};
            float pred = net.predict(input)[0];
            uniquePreds.add(String.format("%.6f", pred));
        }
        
        System.out.printf("  Final: %d unique predictions %s\n\n", 
            uniquePreds.size(), 
            uniquePreds.size() > 3 ? "✓ STABLE" : "⚠️ COLLAPSED");
    }
    
    // Dummy method since we can't easily access internal step count
    private int getTrainingStepCount(NeuralNet net) {
        // This is a placeholder - in reality we'd track steps manually
        return 0;
    }
}