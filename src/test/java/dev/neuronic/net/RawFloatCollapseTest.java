package dev.neuronic.net;

import dev.neuronic.net.optimizers.AdamWOptimizer;
import dev.neuronic.net.optimizers.SgdOptimizer;
import org.junit.jupiter.api.Test;

import java.util.*;

/**
 * Test with raw float inputs (no MixedFeatureInputLayer) to isolate the collapse issue.
 * TODO: REMOVE THIS FILE AFTER DEBUGGING
 */
public class RawFloatCollapseTest {
    
    @Test
    public void testRawFloatInputs() {
        System.out.println("=== RAW FLOAT INPUT COLLAPSE TEST ===\n");
        
        // Test 1: Simplest possible - raw floats directly to output (no hidden layers)
        testRawFloatNetwork("DIRECT_TO_OUTPUT", () -> {
            return NeuralNet.newBuilder()
                .input(3) // 3 raw float inputs
                .setDefaultOptimizer(new SgdOptimizer(0.01f))
                .output(Layers.outputLinearRegression(1)); // No hidden layers
        });
        
        // Test 2: Raw floats with single hidden layer
        testRawFloatNetwork("SINGLE_HIDDEN", () -> {
            return NeuralNet.newBuilder()
                .input(3)
                .setDefaultOptimizer(new SgdOptimizer(0.01f))
                .layer(Layers.hiddenDenseRelu(16))
                .output(Layers.outputLinearRegression(1));
        });
        
        // Test 3: Raw floats with AdamW
        testRawFloatNetwork("ADAMW_HIDDEN", () -> {
            return NeuralNet.newBuilder()
                .input(3)
                .setDefaultOptimizer(new AdamWOptimizer(0.01f, 0.001f))
                .layer(Layers.hiddenDenseRelu(32))
                .layer(Layers.hiddenDenseRelu(16))
                .output(Layers.outputLinearRegression(1));
        });
        
        // Test 4: Linear activation (no ReLU)
        testRawFloatNetwork("LINEAR_ACTIVATION", () -> {
            return NeuralNet.newBuilder()
                .input(3)
                .setDefaultOptimizer(new SgdOptimizer(0.01f))
                .layer(Layers.hiddenDenseLinear(16))
                .output(Layers.outputLinearRegression(1));
        });
    }
    
    @Test
    public void testPredictionValues() {
        System.out.println("=== PREDICTION VALUE ANALYSIS ===\n");
        
        // Create simple network and log actual prediction values
        NeuralNet net = NeuralNet.newBuilder()
            .input(2)
            .setDefaultOptimizer(new SgdOptimizer(0.01f))
            .layer(Layers.hiddenDenseRelu(8))
            .output(Layers.outputLinearRegression(1));
        
        Random rand = new Random(42);
        
        System.out.println("Initial predictions (before training):");
        for (int i = 0; i < 10; i++) {
            float[] input = {rand.nextFloat(), rand.nextFloat()};
            float pred = net.predict(input)[0];
            System.out.printf("  Input [%.3f, %.3f] → %.6f\n", input[0], input[1], pred);
        }
        
        System.out.println("\nTraining with diverse patterns...");
        
        // Train with very diverse patterns
        for (int step = 0; step < 500; step++) {
            float[] input = {rand.nextFloat() * 10, rand.nextFloat() * 10}; // Wide range
            float target = (input[0] > input[1]) ? 1.0f : 0.0f; // Clear pattern
            
            net.trainBatch(new float[][]{input}, new float[][]{new float[]{target}});
        }
        
        System.out.println("\nPredictions after training:");
        Set<String> uniquePreds = new HashSet<>();
        float[] testInputs = {
            0.1f, 0.2f,  // input[0] < input[1] → should predict ~0
            0.3f, 0.4f,
            0.2f, 0.1f,  // input[0] > input[1] → should predict ~1
            0.4f, 0.3f,
            0.5f, 0.5f,  // Equal → unclear
            1.0f, 2.0f,  // Clear 0 case
            2.0f, 1.0f,  // Clear 1 case
            5.0f, 3.0f   // Strong 1 case
        };
        
        for (int i = 0; i < testInputs.length; i += 2) {
            float[] input = {testInputs[i], testInputs[i+1]};
            float pred = net.predict(input)[0];
            float expected = (input[0] > input[1]) ? 1.0f : 0.0f;
            uniquePreds.add(String.format("%.6f", pred));
            
            System.out.printf("  Input [%.1f, %.1f] → %.6f (expected %.1f, diff %.3f)\n", 
                input[0], input[1], pred, expected, Math.abs(pred - expected));
        }
        
        System.out.printf("\nUnique predictions: %d\n", uniquePreds.size());
        System.out.println("All prediction values:");
        for (String pred : uniquePreds) {
            System.out.println("  " + pred);
        }
        
        boolean collapsed = uniquePreds.size() < 5;
        System.out.printf("Result: %s\n\n", collapsed ? "⚠️ COLLAPSED" : "✓ LEARNING");
    }
    
    private void testRawFloatNetwork(String name, NetworkBuilder builder) {
        System.out.println("--- " + name + " ---");
        
        try {
            NeuralNet net = builder.build();
            
            Random rand = new Random(42);
            
            // Train with clear pattern: if input[0] > input[1], target = 1, else target = 0
            for (int step = 0; step < 1000; step++) {
                float[] input = {rand.nextFloat() * 2, rand.nextFloat() * 2, rand.nextFloat()};
                float target = (input[0] > input[1]) ? 1.0f : 0.0f;
                
                net.trainBatch(new float[][]{input}, new float[][]{new float[]{target}});
            }
            
            // Test predictions
            Set<String> uniquePreds = new HashSet<>();
            float trueSum = 0, falseSum = 0;
            int trueCount = 0, falseCount = 0;
            
            for (int i = 0; i < 100; i++) {
                // Create clear test cases
                float[] input;
                boolean shouldBeTrue;
                
                if (i < 50) {
                    // Cases where input[0] > input[1] (should predict ~1)
                    input = new float[]{1.5f, 0.5f, 0.5f};
                    shouldBeTrue = true;
                } else {
                    // Cases where input[0] < input[1] (should predict ~0)
                    input = new float[]{0.5f, 1.5f, 0.5f};
                    shouldBeTrue = false;
                }
                
                float pred = net.predict(input)[0];
                uniquePreds.add(String.format("%.4f", pred));
                
                if (shouldBeTrue) {
                    trueSum += pred;
                    trueCount++;
                } else {
                    falseSum += pred;
                    falseCount++;
                }
            }
            
            float trueAvg = trueSum / trueCount;
            float falseAvg = falseSum / falseCount;
            boolean collapsed = uniquePreds.size() < 5;
            
            System.out.printf("Unique predictions: %d\n", uniquePreds.size());
            System.out.printf("True avg: %.4f, False avg: %.4f\n", trueAvg, falseAvg);
            System.out.printf("Discrimination: %.4f\n", Math.abs(trueAvg - falseAvg));
            System.out.printf("Result: %s\n\n", collapsed ? "⚠️ COLLAPSED" : "✓ LEARNING");
            
        } catch (Exception e) {
            System.out.printf("ERROR: %s\n\n", e.getMessage());
        }
    }
    
    @FunctionalInterface
    interface NetworkBuilder {
        NeuralNet build();
    }
}