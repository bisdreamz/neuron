package dev.neuronic.net;

import dev.neuronic.net.layers.Feature;
import dev.neuronic.net.optimizers.AdamOptimizer;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import dev.neuronic.net.optimizers.SgdOptimizer;
import dev.neuronic.net.simple.SimpleNet;
import dev.neuronic.net.simple.SimpleNetFloat;
import org.junit.jupiter.api.Test;

import java.util.*;

/**
 * Test different optimizers and loss functions to isolate collapse cause.
 * TODO: REMOVE THIS FILE AFTER DEBUGGING
 */
public class OptimizerLossCollapseTest {
    
    @Test
    public void testDifferentOptimizers() {
        System.out.println("=== OPTIMIZER COLLAPSE TEST ===\n");
        
        // Test SGD (simplest optimizer)
        testOptimizer(new SgdOptimizer(0.01f), "SGD", "LINEAR_MSE");
        
        // Test Adam (no weight decay)
        testOptimizer(new AdamOptimizer(0.01f), "ADAM", "LINEAR_MSE");
        
        // Test AdamW (with weight decay - current failing case)
        testOptimizer(new AdamWOptimizer(0.01f, 0.001f), "ADAMW", "LINEAR_MSE");
        
        // Test AdamW with no weight decay
        testOptimizer(new AdamWOptimizer(0.01f, 0.0f), "ADAMW_NO_DECAY", "LINEAR_MSE");
    }
    
    @Test
    public void testDifferentLossFunctions() {
        System.out.println("=== LOSS FUNCTION COLLAPSE TEST ===\n");
        
        // Linear regression (MSE)
        testLossFunction("LINEAR_MSE");
        
        // Huber regression (more robust to outliers)
        testLossFunction("HUBER");
    }
    
    private void testOptimizer(dev.neuronic.net.optimizers.Optimizer optimizer, String optimizerName, String lossType) {
        System.out.println("--- " + optimizerName + " with " + lossType + " ---");
        
        // Simple embedding to isolate optimizer issue
        Feature[] features = {Feature.embedding(1000, 16, "item")};
        
        NeuralNet net;
        if (lossType.equals("HUBER")) {
            net = NeuralNet.newBuilder()
                .input(features.length)
                .setDefaultOptimizer(optimizer)
                .layer(Layers.inputMixed(features))
                .layer(Layers.hiddenDenseRelu(64))
                .layer(Layers.hiddenDenseRelu(32))
                .output(Layers.outputHuberRegression(1));
        } else {
            net = NeuralNet.newBuilder()
                .input(features.length)
                .setDefaultOptimizer(optimizer)
                .layer(Layers.inputMixed(features))
                .layer(Layers.hiddenDenseRelu(64))
                .layer(Layers.hiddenDenseRelu(32))
                .output(Layers.outputLinearRegression(1));
        }
        
        testCollapseWithNetwork(net, optimizerName + "_" + lossType);
    }
    
    private void testLossFunction(String lossType) {
        System.out.println("--- AdamW with " + lossType + " ---");
        
        AdamWOptimizer optimizer = new AdamWOptimizer(0.01f, 0.001f);
        Feature[] features = {Feature.embedding(1000, 16, "item")};
        
        NeuralNet net;
        if (lossType.equals("HUBER")) {
            net = NeuralNet.newBuilder()
                .input(features.length)
                .setDefaultOptimizer(optimizer)
                .layer(Layers.inputMixed(features))
                .layer(Layers.hiddenDenseRelu(64))
                .layer(Layers.hiddenDenseRelu(32))
                .output(Layers.outputHuberRegression(1));
        } else {
            net = NeuralNet.newBuilder()
                .input(features.length)
                .setDefaultOptimizer(optimizer)
                .layer(Layers.inputMixed(features))
                .layer(Layers.hiddenDenseRelu(64))
                .layer(Layers.hiddenDenseRelu(32))
                .output(Layers.outputLinearRegression(1));
        }
        
        testCollapseWithNetwork(net, "ADAMW_" + lossType);
    }
    
    private void testCollapseWithNetwork(NeuralNet net, String testName) {
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        Random rand = new Random(42);
        int trainedSamples = 0;
        
        // Sparse training with good/bad pattern (same as production)
        for (int step = 0; step < 2000; step++) {
            boolean isGood = rand.nextBoolean();
            
            Map<String, Object> input = new HashMap<>();
            if (isGood) {
                input.put("item", "good_" + rand.nextInt(50));
            } else {
                input.put("item", "bad_" + rand.nextInt(500));
            }
            
            // Sparse training (2% like production)
            if (rand.nextFloat() < 0.02f) {
                float target = isGood ? 1.0f : -0.5f;
                model.train(input, target);
                trainedSamples++;
            }
        }
        
        // Test for collapse
        Set<String> uniquePreds = new HashSet<>();
        float goodSum = 0, badSum = 0;
        int goodCount = 0, badCount = 0;
        
        for (int i = 0; i < 100; i++) {
            Map<String, Object> input = new HashMap<>();
            boolean isGood = i < 50;
            
            if (isGood) {
                input.put("item", "good_0");
            } else {
                input.put("item", "bad_999");
            }
            
            float pred = model.predictFloat(input);
            uniquePreds.add(String.format("%.3f", pred));
            
            if (isGood) {
                goodSum += pred;
                goodCount++;
            } else {
                badSum += pred;
                badCount++;
            }
        }
        
        float goodAvg = goodSum / goodCount;
        float badAvg = badSum / badCount;
        boolean collapsed = uniquePreds.size() < 5;
        
        System.out.printf("Trained %d samples\\n", trainedSamples);
        System.out.printf("Unique predictions: %d %s\\n", uniquePreds.size(), 
            collapsed ? "⚠️ COLLAPSED!" : "✓ OK");
        System.out.printf("Good avg: %.3f, Bad avg: %.3f\\n", goodAvg, badAvg);
        System.out.printf("Discrimination: %.3f\\n", goodAvg - badAvg);
        
        // Log some actual predictions to see the pattern
        System.out.print("Sample predictions: ");
        int count = 0;
        for (String pred : uniquePreds) {
            if (count++ >= 5) break;
            System.out.print(pred + " ");
        }
        System.out.println();
        System.out.println();
    }
}