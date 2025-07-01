package dev.neuronic.net;

import dev.neuronic.net.layers.Feature;
import dev.neuronic.net.layers.Layer;
import dev.neuronic.net.layers.MixedFeatureInputLayer;
import dev.neuronic.net.layers.DenseLayer;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import dev.neuronic.net.simple.SimpleNet;
import dev.neuronic.net.simple.SimpleNetFloat;
import org.junit.jupiter.api.Test;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Deep diagnostic test to find the root cause of mode collapse.
 */
public class DeepDiagnosticTest {
    
    @Test
    public void testEmbeddingWeightUpdates() {
        System.out.println("=== TESTING IF EMBEDDINGS ARE ACTUALLY UPDATING ===\n");
        
        Feature[] features = {
            Feature.embedding(10, 8, "item"),  // Small embedding for easy tracking
            Feature.oneHot(3, "category")
        };
        
        AdamWOptimizer optimizer = new AdamWOptimizer(0.1f, 0.0f); // High LR to see changes
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(features.length)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(features))
            .layer(Layers.hiddenDenseRelu(16))
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        // Get initial prediction for item 0
        Map<String, Object> input0 = new HashMap<>();
        input0.put("item", 0);
        input0.put("category", "A");
        
        float pred0_before = model.predictFloat(input0);
        
        // Train ONLY on item 0 with high target
        for (int i = 0; i < 100; i++) {
            model.train(input0, 10.0f);
        }
        
        float pred0_after = model.predictFloat(input0);
        
        // Check item 1 (should not change much)
        Map<String, Object> input1 = new HashMap<>();
        input1.put("item", 1);
        input1.put("category", "A");
        
        float pred1_before = model.predictFloat(input1);
        
        System.out.printf("Item 0: %.3f -> %.3f (change: %.3f)\n", 
            pred0_before, pred0_after, pred0_after - pred0_before);
        System.out.printf("Item 1: %.3f (should be relatively unchanged)\n", pred1_before);
        
        float change = Math.abs(pred0_after - pred0_before);
        assertTrue(change > 1.0f, "Embedding for item 0 should have changed significantly");
        
        // Now test with mixed positive/negative training
        System.out.println("\n=== MIXED POSITIVE/NEGATIVE TRAINING ===");
        
        // Reset with new model
        NeuralNet net2 = NeuralNet.newBuilder()
            .input(features.length)
            .setDefaultOptimizer(new AdamWOptimizer(0.01f, 0.0f))
            .layer(Layers.inputMixed(features))
            .layer(Layers.hiddenDenseRelu(32))
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat model2 = SimpleNet.ofFloatRegression(net2);
        
        // Track predictions over time
        List<Float> avgPredictions = new ArrayList<>();
        Random rand = new Random(42);
        
        for (int epoch = 0; epoch < 20; epoch++) {
            // Train with 1% positive, 2% negative
            for (int i = 0; i < 1000; i++) {
                Map<String, Object> input = new HashMap<>();
                input.put("item", rand.nextInt(10));
                input.put("category", rand.nextInt(3));
                
                if (rand.nextFloat() < 0.01f) {
                    model2.train(input, 1.0f + rand.nextFloat() * 2.0f);
                } else if (rand.nextFloat() < 0.02f) {
                    model2.train(input, -0.01f);
                }
            }
            
            // Check average prediction
            float sum = 0;
            for (int i = 0; i < 100; i++) {
                Map<String, Object> input = new HashMap<>();
                input.put("item", rand.nextInt(10));
                input.put("category", rand.nextInt(3));
                sum += model2.predictFloat(input);
            }
            avgPredictions.add(sum / 100);
            
            System.out.printf("Epoch %2d: Avg prediction = %.3f\n", epoch, sum / 100);
        }
        
        // Check if predictions are converging
        float firstAvg = avgPredictions.get(0);
        float lastAvg = avgPredictions.get(avgPredictions.size() - 1);
        System.out.printf("\nPrediction drift: %.3f -> %.3f\n", firstAvg, lastAvg);
    }
    
    @Test
    public void testFinalLayerBiasDominance() {
        System.out.println("\n=== TESTING FINAL LAYER BIAS DOMINANCE ===\n");
        
        Feature[] features = {
            Feature.embedding(100, 16, "item"),
            Feature.oneHot(10, "category")
        };
        
        AdamWOptimizer optimizer = new AdamWOptimizer(0.01f, 0.001f);
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(features.length)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(features))
            .layer(Layers.hiddenDenseRelu(64))
            .layer(Layers.hiddenDenseRelu(32))
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        Random rand = new Random(42);
        
        // Get initial predictions to see range
        float initSum = 0, initMin = Float.MAX_VALUE, initMax = Float.MIN_VALUE;
        for (int i = 0; i < 100; i++) {
            Map<String, Object> input = new HashMap<>();
            input.put("item", rand.nextInt(100));
            input.put("category", rand.nextInt(10));
            float pred = model.predictFloat(input);
            initSum += pred;
            initMin = Math.min(initMin, pred);
            initMax = Math.max(initMax, pred);
        }
        
        System.out.printf("Initial predictions: avg=%.3f, range=[%.3f, %.3f]\n", 
            initSum/100, initMin, initMax);
        
        // Train with sparse signals
        for (int i = 0; i < 10000; i++) {
            Map<String, Object> input = new HashMap<>();
            input.put("item", rand.nextInt(100));
            input.put("category", rand.nextInt(10));
            
            if (rand.nextFloat() < 0.01f) {
                model.train(input, 1.0f + rand.nextFloat());
            } else if (rand.nextFloat() < 0.02f) {
                model.train(input, -0.01f);
            }
        }
        
        // Check final predictions
        float finalSum = 0, finalMin = Float.MAX_VALUE, finalMax = Float.MIN_VALUE;
        Set<String> uniquePreds = new HashSet<>();
        
        for (int i = 0; i < 100; i++) {
            Map<String, Object> input = new HashMap<>();
            input.put("item", rand.nextInt(100));
            input.put("category", rand.nextInt(10));
            float pred = model.predictFloat(input);
            finalSum += pred;
            finalMin = Math.min(finalMin, pred);
            finalMax = Math.max(finalMax, pred);
            uniquePreds.add(String.format("%.3f", pred));
        }
        
        System.out.printf("Final predictions: avg=%.3f, range=[%.3f, %.3f]\n", 
            finalSum/100, finalMin, finalMax);
        System.out.printf("Unique predictions: %d/100\n", uniquePreds.size());
        
        // Check if predictions collapsed
        if (uniquePreds.size() < 10) {
            System.out.println("⚠️  WARNING: Predictions have collapsed to few unique values!");
        }
        
        float rangeRatio = (finalMax - finalMin) / (initMax - initMin);
        System.out.printf("Range reduction: %.1f%%\n", (1 - rangeRatio) * 100);
    }
    
    @Test
    public void testGradientMagnitudes() {
        System.out.println("\n=== TESTING GRADIENT MAGNITUDES ===\n");
        
        Feature[] features = {
            Feature.embedding(10, 8, "item")
        };
        
        // Test with different learning rates
        for (float lr : new float[]{0.001f, 0.01f, 0.1f}) {
            System.out.printf("\nTesting with learning rate %.3f:\n", lr);
            
            AdamWOptimizer optimizer = new AdamWOptimizer(lr, 0.0f);
            
            NeuralNet net = NeuralNet.newBuilder()
                .input(features.length)
                .setDefaultOptimizer(optimizer)
                .layer(Layers.inputMixed(features))
                .layer(Layers.hiddenDenseRelu(16))
                .output(Layers.outputLinearRegression(1));
                
            SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
            
            // Train with extreme sparse signals
            Random rand = new Random(42);
            float sumPred = 0;
            int predCount = 0;
            
            for (int i = 0; i < 5000; i++) {
                Map<String, Object> input = new HashMap<>();
                input.put("item", rand.nextInt(10));
                
                if (rand.nextFloat() < 0.005f) { // 0.5% positive
                    model.train(input, 2.0f);
                } else if (rand.nextFloat() < 0.02f) { // 2% negative
                    model.train(input, -0.01f);
                }
                
                // Sample predictions
                if (i % 1000 == 0 && i > 0) {
                    float pred = model.predictFloat(input);
                    sumPred += pred;
                    predCount++;
                    System.out.printf("  Step %d: prediction = %.3f\n", i, pred);
                }
            }
            
            float avgPred = sumPred / predCount;
            System.out.printf("  Average prediction: %.3f\n", avgPred);
        }
    }
    
    @Test  
    public void testEmbeddingInitialization() {
        System.out.println("\n=== TESTING EMBEDDING INITIALIZATION IMPACT ===\n");
        
        // The embeddings should be initialized with uniform[-0.05, 0.05]
        // Let's verify this is happening
        
        Feature[] features = {
            Feature.embedding(1000, 32, "item"),  // Large embedding table
            Feature.oneHot(10, "category")
        };
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(features.length)
            .setDefaultOptimizer(new AdamWOptimizer(0.01f, 0.0f))
            .layer(Layers.inputMixed(features))
            .layer(Layers.hiddenDenseRelu(64))
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        // Check initial predictions - should be near zero with small uniform init
        Random rand = new Random(42);
        float sum = 0;
        float min = Float.MAX_VALUE;
        float max = Float.MIN_VALUE;
        
        for (int i = 0; i < 1000; i++) {
            Map<String, Object> input = new HashMap<>();
            input.put("item", rand.nextInt(1000));
            input.put("category", rand.nextInt(10));
            
            float pred = model.predictFloat(input);
            sum += pred;
            min = Math.min(min, pred);
            max = Math.max(max, pred);
        }
        
        float avg = sum / 1000;
        System.out.printf("Initial predictions: avg=%.3f, range=[%.3f, %.3f]\n", avg, min, max);
        
        // If initialization is too small, predictions will be too similar
        float range = max - min;
        if (range < 0.1f) {
            System.out.println("⚠️  WARNING: Initial predictions have very small range!");
            System.out.println("   This could lead to mode collapse as gradients are too similar.");
        }
    }
    
    @Test
    public void testSimplestCase() {
        System.out.println("\n=== SIMPLEST POSSIBLE CASE ===\n");
        
        // Just one embedding, no other features
        Feature[] features = {
            Feature.embedding(5, 4, "item")  // 5 items, 4-dim embeddings
        };
        
        AdamWOptimizer optimizer = new AdamWOptimizer(0.1f, 0.0f); // High LR
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(features.length)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(features))
            .layer(Layers.hiddenDenseRelu(8))
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        // Train each item to a different target
        Map<Integer, Float> targets = new HashMap<>();
        targets.put(0, 0.1f);
        targets.put(1, 0.5f);
        targets.put(2, 1.0f);
        targets.put(3, 1.5f);
        targets.put(4, 2.0f);
        
        // Initial predictions
        System.out.println("Initial predictions:");
        for (int item = 0; item < 5; item++) {
            Map<String, Object> input = new HashMap<>();
            input.put("item", item);
            float pred = model.predictFloat(input);
            System.out.printf("  Item %d: %.3f (target: %.1f)\n", item, pred, targets.get(item));
        }
        
        // Train
        for (int epoch = 0; epoch < 100; epoch++) {
            for (Map.Entry<Integer, Float> entry : targets.entrySet()) {
                Map<String, Object> input = new HashMap<>();
                input.put("item", entry.getKey());
                model.train(input, entry.getValue());
            }
        }
        
        // Final predictions
        System.out.println("\nFinal predictions:");
        boolean canLearn = true;
        for (int item = 0; item < 5; item++) {
            Map<String, Object> input = new HashMap<>();
            input.put("item", item);
            float pred = model.predictFloat(input);
            float target = targets.get(item);
            float error = Math.abs(pred - target);
            System.out.printf("  Item %d: %.3f (target: %.1f, error: %.3f)\n", 
                item, pred, target, error);
            
            if (error > 0.5f) {
                canLearn = false;
            }
        }
        
        if (!canLearn) {
            System.out.println("\n❌ CRITICAL: Network cannot learn even simple distinct mappings!");
        } else {
            System.out.println("\n✓ Network can learn distinct values in simple case");
        }
    }
}