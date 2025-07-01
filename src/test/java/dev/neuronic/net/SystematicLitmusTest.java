package dev.neuronic.net;

import dev.neuronic.net.layers.Feature;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import dev.neuronic.net.optimizers.SgdOptimizer;
import dev.neuronic.net.simple.SimpleNet;
import dev.neuronic.net.simple.SimpleNetFloat;
import org.junit.jupiter.api.Test;

import java.util.*;

/**
 * Systematic litmus tests to isolate the exact cause of mode collapse.
 * TODO: REMOVE THIS FILE AFTER DEBUGGING
 */
public class SystematicLitmusTest {
    
    @Test
    public void testArchitectureVariations() {
        System.out.println("=== ARCHITECTURE VARIATION TESTS ===\n");
        
        // Test 1: Simplest possible - embedding directly to output
        testArchitecture("EMBEDDING_TO_OUTPUT", () -> {
            Feature[] features = {Feature.embedding(100, 8, "item")};
            return NeuralNet.newBuilder()
                .input(features.length)
                .setDefaultOptimizer(new SgdOptimizer(0.1f))
                .layer(Layers.inputMixed(features))
                .output(Layers.outputLinearRegression(1)); // NO hidden layers
        });
        
        // Test 2: Passthrough features only (no embeddings)
        testArchitecture("PASSTHROUGH_ONLY", () -> {
            Feature[] features = {
                Feature.passthrough("value1"),
                Feature.passthrough("value2"),
                Feature.passthrough("value3")
            };
            return NeuralNet.newBuilder()
                .input(features.length)
                .setDefaultOptimizer(new SgdOptimizer(0.1f))
                .layer(Layers.inputMixed(features))
                .layer(Layers.hiddenDenseRelu(32))
                .output(Layers.outputLinearRegression(1));
        });
        
        // Test 3: Traditional dense layers only (no mixed features)
        testArchitecture("DENSE_ONLY", () -> {
            return NeuralNet.newBuilder()
                .input(3) // 3 raw float inputs
                .setDefaultOptimizer(new SgdOptimizer(0.1f))
                .layer(Layers.hiddenDenseRelu(32))
                .layer(Layers.hiddenDenseRelu(16))
                .output(Layers.outputLinearRegression(1));
        });
        
        // Test 4: Single hidden layer
        testArchitecture("SINGLE_HIDDEN", () -> {
            Feature[] features = {Feature.embedding(100, 8, "item")};
            return NeuralNet.newBuilder()
                .input(features.length)
                .setDefaultOptimizer(new SgdOptimizer(0.1f))
                .layer(Layers.inputMixed(features))
                .layer(Layers.hiddenDenseRelu(32)) // Single hidden layer
                .output(Layers.outputLinearRegression(1));
        });
        
        // Test 5: Linear activation (no ReLU)
        testArchitecture("LINEAR_ACTIVATION", () -> {
            Feature[] features = {Feature.embedding(100, 8, "item")};
            return NeuralNet.newBuilder()
                .input(features.length)
                .setDefaultOptimizer(new SgdOptimizer(0.1f))
                .layer(Layers.inputMixed(features))
                .layer(Layers.hiddenDenseLinear(32)) // Linear instead of ReLU
                .output(Layers.outputLinearRegression(1));
        });
    }
    
    @Test
    public void testTrainingMechanics() {
        System.out.println("=== TRAINING MECHANICS TESTS ===\n");
        
        // Test different batch sizes vs single-sample
        testBatchTraining("SINGLE_SAMPLE", 1);
        testBatchTraining("SMALL_BATCH", 8);
        testBatchTraining("LARGE_BATCH", 32);
        
        // Test different learning rates
        testLearningRate("LR_0001", 0.001f);
        testLearningRate("LR_01", 0.01f);
        testLearningRate("LR_1", 0.1f);
        testLearningRate("LR_VERY_HIGH", 1.0f);
    }
    
    @Test
    public void testTargetValueRanges() {
        System.out.println("=== TARGET VALUE RANGE TESTS ===\n");
        
        // Test different target ranges
        testTargetRange("SMALL_RANGE", 0.0f, 1.0f);      // 0-1
        testTargetRange("MEDIUM_RANGE", -1.0f, 1.0f);    // -1 to 1  
        testTargetRange("LARGE_RANGE", -10.0f, 10.0f);   // -10 to 10
        testTargetRange("POSITIVE_ONLY", 0.1f, 5.0f);    // Positive only
    }
    
    private void testArchitecture(String name, NetworkBuilder builder) {
        System.out.println("--- " + name + " ---");
        
        try {
            NeuralNet net = builder.build();
            boolean useMixed = name.contains("EMBEDDING") || name.contains("PASSTHROUGH");
            boolean collapsed = testNetworkForCollapse(net, name, useMixed);
            
            System.out.printf("Result: %s\\n\\n", collapsed ? "⚠️ COLLAPSED" : "✓ WORKING");
        } catch (Exception e) {
            System.out.printf("ERROR: %s\\n\\n", e.getMessage());
        }
    }
    
    private void testBatchTraining(String name, int batchSize) {
        System.out.println("--- " + name + " (batch size " + batchSize + ") ---");
        
        Feature[] features = {Feature.embedding(100, 8, "item")};
        NeuralNet net = NeuralNet.newBuilder()
            .input(features.length)
            .setDefaultOptimizer(new SgdOptimizer(0.1f))
            .layer(Layers.inputMixed(features))
            .layer(Layers.hiddenDenseRelu(32))
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        // Train with specified batch size
        Random rand = new Random(42);
        List<Map<String, Object>> batchInputs = new ArrayList<>();
        List<Float> batchTargets = new ArrayList<>();
        
        for (int i = 0; i < 1000; i++) {
            boolean isGood = rand.nextBoolean();
            Map<String, Object> input = Map.of("item", isGood ? "good_" + rand.nextInt(20) : "bad_" + rand.nextInt(80));
            float target = isGood ? 1.0f : 0.0f;
            
            batchInputs.add(input);
            batchTargets.add(target);
            
            // Train when batch is full
            if (batchInputs.size() == batchSize) {
                for (int j = 0; j < batchInputs.size(); j++) {
                    model.train(batchInputs.get(j), batchTargets.get(j));
                }
                batchInputs.clear();
                batchTargets.clear();
            }
        }
        
        // Test for collapse
        boolean collapsed = checkCollapse(model, true);
        System.out.printf("Result: %s\\n\\n", collapsed ? "⚠️ COLLAPSED" : "✓ WORKING");
    }
    
    private void testLearningRate(String name, float lr) {
        System.out.println("--- " + name + " (LR=" + lr + ") ---");
        
        Feature[] features = {Feature.embedding(50, 8, "item")};
        NeuralNet net = NeuralNet.newBuilder()
            .input(features.length)
            .setDefaultOptimizer(new SgdOptimizer(lr))
            .layer(Layers.inputMixed(features))
            .layer(Layers.hiddenDenseRelu(32))
            .output(Layers.outputLinearRegression(1));
            
        boolean collapsed = testNetworkForCollapse(net, name, true);
        System.out.printf("Result: %s\\n\\n", collapsed ? "⚠️ COLLAPSED" : "✓ WORKING");
    }
    
    private void testTargetRange(String name, float minTarget, float maxTarget) {
        System.out.println("--- " + name + " (targets " + minTarget + " to " + maxTarget + ") ---");
        
        Feature[] features = {Feature.embedding(50, 8, "item")};
        NeuralNet net = NeuralNet.newBuilder()
            .input(features.length)
            .setDefaultOptimizer(new SgdOptimizer(0.1f))
            .layer(Layers.inputMixed(features))
            .layer(Layers.hiddenDenseRelu(32))
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        Random rand = new Random(42);
        
        // Train with specified target range
        for (int i = 0; i < 1000; i++) {
            boolean isGood = rand.nextBoolean();
            Map<String, Object> input = Map.of("item", isGood ? "good_" + rand.nextInt(20) : "bad_" + rand.nextInt(80));
            float target = isGood ? 
                (minTarget + rand.nextFloat() * (maxTarget - minTarget)) : 
                minTarget;
            model.train(input, target);
        }
        
        boolean collapsed = checkCollapse(model, true);
        System.out.printf("Result: %s\\n\\n", collapsed ? "⚠️ COLLAPSED" : "✓ WORKING");
    }
    
    private boolean testNetworkForCollapse(NeuralNet net, String name, boolean useMixed) {
        if (useMixed) {
            SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
            
            Random rand = new Random(42);
            for (int i = 0; i < 1000; i++) {
                boolean isGood = rand.nextBoolean();
                Map<String, Object> input = Map.of("item", isGood ? "good_" + rand.nextInt(20) : "bad_" + rand.nextInt(80));
                float target = isGood ? 1.0f : 0.0f;
                model.train(input, target);
            }
            
            return checkCollapse(model, true);
        } else {
            // Raw float array training for dense-only networks
            Random rand = new Random(42);
            for (int i = 0; i < 1000; i++) {
                boolean isGood = rand.nextBoolean();
                float[] input = {isGood ? 1.0f : 0.0f, rand.nextFloat(), rand.nextFloat()};
                float[] target = {isGood ? 1.0f : 0.0f};
                net.trainBatch(new float[][]{input}, new float[][]{target});
            }
            
            // Test predictions
            Set<String> uniquePreds = new HashSet<>();
            for (int i = 0; i < 20; i++) {
                float[] input = {i < 10 ? 1.0f : 0.0f, 0.5f, 0.5f};
                float pred = net.predict(input)[0];
                uniquePreds.add(String.format("%.3f", pred));
            }
            
            return uniquePreds.size() < 3;
        }
    }
    
    private boolean checkCollapse(SimpleNetFloat model, boolean useMixed) {
        Set<String> uniquePreds = new HashSet<>();
        Random rand = new Random(123);
        
        for (int i = 0; i < 50; i++) {
            Map<String, Object> input = Map.of("item", i < 25 ? "good_0" : "bad_999");
            float pred = model.predictFloat(input);
            uniquePreds.add(String.format("%.3f", pred));
        }
        
        return uniquePreds.size() < 3; // Collapsed if less than 3 unique predictions
    }
    
    @FunctionalInterface
    interface NetworkBuilder {
        NeuralNet build();
    }
}