package dev.neuronic.net;

import dev.neuronic.net.layers.Feature;
import dev.neuronic.net.optimizers.SgdOptimizer;
import dev.neuronic.net.simple.SimpleNet;
import dev.neuronic.net.simple.SimpleNetFloat;
import org.junit.jupiter.api.Test;

import java.util.*;

/**
 * Explains why all predictions change when training one embedding.
 * TODO: REMOVE THIS FILE AFTER DEBUGGING
 */
public class OutputLayerExplanationTest {
    
    @Test
    public void explainWhyAllPredictionsChange() {
        System.out.println("=== WHY ALL PREDICTIONS CHANGE ===\n");
        
        Feature[] features = {
            Feature.embedding(3, 1, "item")  // 3 items, 1-dim embeddings
        };
        
        SgdOptimizer optimizer = new SgdOptimizer(0.1f);
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(features.length)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(features))
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        System.out.println("Network structure:");
        System.out.println("  Embedding(3,1) -> Linear(1,1) -> Output");
        System.out.println("  Output = W * embedding + b");
        System.out.println();
        
        // Initial predictions
        System.out.println("Initial predictions:");
        Map<Integer, Float> initialPreds = new HashMap<>();
        for (int i = 0; i < 3; i++) {
            Map<String, Object> input = new HashMap<>();
            input.put("item", i);
            float pred = model.predictFloat(input);
            initialPreds.put(i, pred);
            System.out.printf("  Item %d: %.3f = W * emb[%d] + b\n", i, pred, i);
        }
        
        // Train item 0
        System.out.println("\nTraining item 0 -> 1.0");
        Map<String, Object> input0 = new HashMap<>();
        input0.put("item", 0);
        model.train(input0, 1.0f);
        
        // What changed:
        System.out.println("\nWhat changed during training:");
        System.out.println("  1. Embedding[0] changed (via gradient)");
        System.out.println("  2. Output weight W changed");
        System.out.println("  3. Output bias b changed");
        
        // New predictions
        System.out.println("\nNew predictions:");
        for (int i = 0; i < 3; i++) {
            Map<String, Object> input = new HashMap<>();
            input.put("item", i);
            float pred = model.predictFloat(input);
            float change = pred - initialPreds.get(i);
            System.out.printf("  Item %d: %.3f (change: %+.3f)\n", i, pred, change);
        }
        
        System.out.println("\nExplanation:");
        System.out.println("  - Item 0 changed because: embedding[0] changed AND W,b changed");
        System.out.println("  - Items 1,2 changed because: W,b changed (even though embeddings[1,2] didn't)");
        System.out.println("\nThis is NORMAL neural network behavior!");
    }
    
    @Test
    public void demonstrateConvergence() {
        System.out.println("\n=== CONVERGENCE WITH DIFFERENT TARGETS ===\n");
        
        Feature[] features = {
            Feature.embedding(5, 8, "item")
        };
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(features.length)
            .setDefaultOptimizer(new SgdOptimizer(0.01f))
            .layer(Layers.inputMixed(features))
            .layer(Layers.hiddenDenseRelu(16))
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        // Train different items to different targets
        float[] targets = {0.1f, 0.3f, 0.5f, 0.7f, 0.9f};
        
        System.out.println("Training 5 items to different targets...");
        for (int epoch = 0; epoch < 100; epoch++) {
            for (int i = 0; i < 5; i++) {
                Map<String, Object> input = new HashMap<>();
                input.put("item", i);
                model.train(input, targets[i]);
            }
        }
        
        System.out.println("\nFinal predictions:");
        for (int i = 0; i < 5; i++) {
            Map<String, Object> input = new HashMap<>();
            input.put("item", i);
            float pred = model.predictFloat(input);
            System.out.printf("  Item %d: %.3f (target: %.1f)\n", i, pred, targets[i]);
        }
        
        System.out.println("\nWith enough hidden units and training, the network CAN learn");
        System.out.println("to map different embeddings to different outputs.");
    }
}