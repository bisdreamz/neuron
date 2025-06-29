package dev.neuronic.net;

import dev.neuronic.net.layers.Feature;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import dev.neuronic.net.simple.SimpleNet;
import dev.neuronic.net.simple.SimpleNetFloat;
import org.junit.jupiter.api.Test;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test to demonstrate what happens when training with a constant target value.
 * This matches the user's scenario where all targets are 1.0199995.
 */
public class ConstantTargetTest {
    
    @Test
    public void testConstantTargetPredictions() {
        System.out.println("=== CONSTANT TARGET TEST ===\n");
        
        AdamWOptimizer optimizer = new AdamWOptimizer(0.01f, 0.0001f);
        
        Feature[] features = {
            Feature.oneHot(200, "country"),
            Feature.oneHot(10, "format"),
            Feature.oneHot(5, "device"),
            Feature.oneHot(10, "os"),
            Feature.oneHot(1000, "pubid"),
            Feature.oneHot(5000, "domain"),
            Feature.oneHot(1000, "zoneid"),
            Feature.oneHot(4, "connection")
        };
        
        NeuralNet net = NeuralNet.newBuilder()
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(features))
            .layer(Layers.hiddenDenseLeakyRelu(128, 0.01f))
            .layer(Layers.hiddenDenseLeakyRelu(64, 0.01f))
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        // Generate varied input data
        Random rand = new Random(42);
        List<Map<String, Object>> inputs = new ArrayList<>();
        
        for (int i = 0; i < 1000; i++) {
            Map<String, Object> input = new HashMap<>();
            input.put("country", rand.nextInt(200));
            input.put("format", rand.nextInt(10));
            input.put("device", rand.nextInt(5));
            input.put("os", rand.nextInt(10));
            input.put("pubid", rand.nextInt(1000));
            input.put("domain", rand.nextInt(5000));
            input.put("zoneid", rand.nextInt(1000));
            input.put("connection", rand.nextInt(4));
            inputs.add(input);
        }
        
        // CRITICAL: All targets are the same value (like user's 1.0199995)
        float constantTarget = 1.02f;
        
        System.out.println("Training with constant target: " + constantTarget);
        System.out.println("Number of unique inputs: " + inputs.size());
        
        // Initial predictions before training
        System.out.println("\nInitial predictions (before training):");
        float initMin = Float.MAX_VALUE, initMax = Float.MIN_VALUE;
        for (int i = 0; i < 10; i++) {
            float pred = model.predictFloat(inputs.get(i));
            System.out.printf("  Sample %d: %.6f\n", i, pred);
            initMin = Math.min(initMin, pred);
            initMax = Math.max(initMax, pred);
        }
        System.out.printf("Initial range: [%.6f, %.6f]\n", initMin, initMax);
        
        // Train for many iterations
        System.out.println("\nTraining for 500 iterations...");
        for (int iter = 0; iter < 500; iter++) {
            for (int i = 0; i < 100; i++) {
                int idx = rand.nextInt(inputs.size());
                model.train(inputs.get(idx), constantTarget);
            }
            
            if (iter % 100 == 0) {
                // Sample predictions during training
                float sum = 0;
                for (int i = 0; i < 10; i++) {
                    sum += model.predictFloat(inputs.get(i));
                }
                System.out.printf("Iter %d: avg prediction = %.6f\n", iter, sum / 10);
            }
        }
        
        // Final predictions after training
        System.out.println("\nFinal predictions (after training):");
        float finalMin = Float.MAX_VALUE, finalMax = Float.MIN_VALUE;
        float sum = 0;
        List<Float> allPredictions = new ArrayList<>();
        
        for (int i = 0; i < 50; i++) {
            float pred = model.predictFloat(inputs.get(i));
            allPredictions.add(pred);
            sum += pred;
            finalMin = Math.min(finalMin, pred);
            finalMax = Math.max(finalMax, pred);
            
            if (i < 10) {
                System.out.printf("  Sample %d: %.6f (target: %.6f, diff: %.6f)\n", 
                                i, pred, constantTarget, Math.abs(pred - constantTarget));
            }
        }
        
        float avgPred = sum / 50;
        System.out.printf("\nFinal statistics:\n");
        System.out.printf("  Average prediction: %.6f\n", avgPred);
        System.out.printf("  Prediction range: [%.6f, %.6f]\n", finalMin, finalMax);
        System.out.printf("  Range width: %.6f\n", finalMax - finalMin);
        System.out.printf("  Target value: %.6f\n", constantTarget);
        
        // Calculate standard deviation
        float variance = 0;
        for (float pred : allPredictions) {
            variance += (pred - avgPred) * (pred - avgPred);
        }
        variance /= allPredictions.size();
        float stdDev = (float) Math.sqrt(variance);
        System.out.printf("  Standard deviation: %.6f\n", stdDev);
        
        System.out.println("\n=== EXPLANATION ===");
        System.out.println("When all targets are the same value:");
        System.out.println("1. The model learns to predict close to that constant");
        System.out.println("2. Small variations come from:");
        System.out.println("   - Random initialization");
        System.out.println("   - Regularization (weight decay)");
        System.out.println("   - Numerical noise");
        System.out.println("3. The model CANNOT learn patterns because there's no signal!");
        System.out.println("\nYour predictions (1.38-1.55) suggest the model is trying to predict ~1.02");
        System.out.println("but with some noise/overfitting to specific feature combinations.");
    }
    
    @Test
    public void testVariedTargets() {
        System.out.println("\n\n=== VARIED TARGETS TEST (for comparison) ===\n");
        
        AdamWOptimizer optimizer = new AdamWOptimizer(0.01f, 0.0001f);
        
        Feature[] features = {
            Feature.oneHot(10, "feature1"),
            Feature.oneHot(10, "feature2"),
            Feature.oneHot(10, "feature3")
        };
        
        NeuralNet net = NeuralNet.newBuilder()
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(features))
            .layer(Layers.hiddenDenseLeakyRelu(32, 0.01f))
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        // Generate data with VARIED targets
        Random rand = new Random(42);
        List<Map<String, Object>> inputs = new ArrayList<>();
        List<Float> targets = new ArrayList<>();
        
        for (int i = 0; i < 100; i++) {
            Map<String, Object> input = new HashMap<>();
            input.put("feature1", i % 10);
            input.put("feature2", (i / 10) % 10);
            input.put("feature3", rand.nextInt(10));
            inputs.add(input);
            
            // VARIED targets based on features
            float target = (i % 10) * 0.1f + ((i / 10) % 10) * 0.2f + rand.nextFloat() * 0.5f;
            targets.add(target);
        }
        
        System.out.println("Training with varied targets...");
        System.out.printf("Target range: [%.2f, %.2f]\n", 
                        targets.stream().min(Float::compare).get(),
                        targets.stream().max(Float::compare).get());
        
        // Train
        for (int epoch = 0; epoch < 50; epoch++) {
            for (int i = 0; i < inputs.size(); i++) {
                model.train(inputs.get(i), targets.get(i));
            }
        }
        
        // Check predictions
        System.out.println("\nPredictions with varied targets:");
        float min = Float.MAX_VALUE, max = Float.MIN_VALUE;
        for (int i = 0; i < 10; i++) {
            float pred = model.predictFloat(inputs.get(i));
            float target = targets.get(i);
            System.out.printf("  Target: %.3f, Prediction: %.3f, Error: %.3f\n", 
                            target, pred, Math.abs(pred - target));
            min = Math.min(min, pred);
            max = Math.max(max, pred);
        }
        
        System.out.printf("\nPrediction range: [%.3f, %.3f]\n", min, max);
        System.out.println("Notice: With varied targets, predictions also vary!");
    }
}