package dev.neuronic.net;

import dev.neuronic.net.layers.Feature;
import dev.neuronic.net.layers.Layer;
import dev.neuronic.net.math.FastRandom;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import dev.neuronic.net.optimizers.Optimizer;
import dev.neuronic.net.optimizers.SgdOptimizer;
import dev.neuronic.net.outputs.RegressionOutput;
import dev.neuronic.net.simple.SimpleNet;
import dev.neuronic.net.simple.SimpleNetFloat;
import org.junit.jupiter.api.Test;

import java.util.*;
import java.util.concurrent.ExecutorService;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test to verify embeddings are actually updating correctly.
 */
public class EmbeddingUpdateTest {

    /**
     * A dummy output layer with no learnable parameters.
     * It simply sums its inputs to produce a single output value.
     * This allows testing the gradient flow to the embedding layer in isolation.
     */
    private static class DummyOutputLayer implements Layer, Layer.Spec, RegressionOutput {
        @Override
        public LayerContext forward(float[] input, boolean isTraining) {
            float sum = 0;
            for (float v : input) {
                sum += v;
            }
            return new LayerContext(input, null, new float[]{sum});
        }

        @Override
        public float[] backward(LayerContext[] stack, int stackIndex, float[] upstreamGradient) {
            // The gradient for each input is just the upstream gradient, since the output is a simple sum.
            float[] downstreamGradient = new float[stack[stackIndex].inputs().length];
            Arrays.fill(downstreamGradient, upstreamGradient[0]);
            return downstreamGradient;
        }

        @Override
        public void applyGradients(float[][] weightGradients, float[] biasGradients) {
            // No-op, no learnable parameters
        }

        @Override
        public GradientDimensions getGradientDimensions() {
            return null; // No learnable parameters for the NeuralNet to manage
        }

        @Override
        public int getOutputSize() {
            return 1;
        }

        @Override
        public Layer create(int inputSize, Optimizer defaultOptimizer, FastRandom random) {
            return this;
        }
    }
    
    @Test
    public void testEmbeddingGradientFlow() {
        System.out.println("=== TESTING EMBEDDING GRADIENT FLOW ===\n");
        
        // Single embedding feature
        Feature[] features = {
            Feature.embedding(3, 4, "item")  // 3 items, 4-dim embeddings
        };
        
        // Use SGD for predictable updates
        SgdOptimizer optimizer = new SgdOptimizer(1.0f); // LR=1 for easy math
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(features.length)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(features))
            .output(new DummyOutputLayer());
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        // Get predictions before training
        Map<String, Object> input0 = new HashMap<>();
        input0.put("item", 0);
        
        Map<String, Object> input1 = new HashMap<>();
        input1.put("item", 1);
        
        Map<String, Object> input2 = new HashMap<>();
        input2.put("item", 2);
        
        System.out.println("Initial predictions:");
        float pred0_init = model.predictFloat(input0);
        float pred1_init = model.predictFloat(input1);
        float pred2_init = model.predictFloat(input2);
        System.out.printf("  Item 0: %.3f\n", pred0_init);
        System.out.printf("  Item 1: %.3f\n", pred1_init);
        System.out.printf("  Item 2: %.3f\n", pred2_init);
        
        // Train only item 0 with target 1.0
        System.out.println("\nTraining item 0 -> 1.0 (single step)");
        model.train(input0, 1.0f);
        
        // Check predictions after one update
        System.out.println("\nAfter 1 training step:");
        float pred0_step1 = model.predictFloat(input0);
        float pred1_step1 = model.predictFloat(input1);
        float pred2_step1 = model.predictFloat(input2);
        System.out.printf("  Item 0: %.3f (was %.3f, change: %.3f)\n", 
            pred0_step1, pred0_init, pred0_step1 - pred0_init);
        System.out.printf("  Item 1: %.3f (was %.3f, change: %.3f)\n", 
            pred1_step1, pred1_init, pred1_step1 - pred1_init);
        System.out.printf("  Item 2: %.3f (was %.3f, change: %.3f)\n", 
            pred2_step1, pred2_init, pred2_step1 - pred2_init);
        
        // Only item 0 should have changed
        assertTrue(Math.abs(pred0_step1 - pred0_init) > 0.01f, 
            "Item 0 should have changed after training");
        assertTrue(Math.abs(pred1_step1 - pred1_init) < 0.001f, 
            "Item 1 should NOT have changed");
        assertTrue(Math.abs(pred2_step1 - pred2_init) < 0.001f, 
            "Item 2 should NOT have changed");
        
        // Train more steps
        for (int i = 0; i < 9; i++) {
            model.train(input0, 1.0f);
        }
        
        System.out.println("\nAfter 10 total training steps on item 0:");
        float pred0_final = model.predictFloat(input0);
        float pred1_final = model.predictFloat(input1);
        float pred2_final = model.predictFloat(input2);
        System.out.printf("  Item 0: %.3f (moving toward 1.0)\n", pred0_final);
        System.out.printf("  Item 1: %.3f (unchanged)\n", pred1_final);
        System.out.printf("  Item 2: %.3f (unchanged)\n", pred2_final);
    }
    
    @Test
    public void testMixedFeatureInteraction() {
        System.out.println("\n=== TESTING MIXED FEATURE INTERACTION ===\n");
        
        // Embedding + one-hot
        Feature[] features = {
            Feature.embedding(3, 4, "item"),
            Feature.oneHot(2, "category")
        };
        
        AdamWOptimizer optimizer = new AdamWOptimizer(0.1f, 0.0f);
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(features.length)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(features))
            .layer(Layers.hiddenDenseRelu(8))
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        // Test if different combinations produce different outputs
        Map<String, Float> predictions = new HashMap<>();
        
        for (int item = 0; item < 3; item++) {
            for (String cat : Arrays.asList("A", "B")) {
                Map<String, Object> input = new HashMap<>();
                input.put("item", item);
                input.put("category", cat);
                
                String key = "item" + item + "_cat" + cat;
                predictions.put(key, model.predictFloat(input));
            }
        }
        
        System.out.println("Initial predictions (should be different):");
        for (Map.Entry<String, Float> entry : predictions.entrySet()) {
            System.out.printf("  %s: %.3f\n", entry.getKey(), entry.getValue());
        }
        
        // Check diversity
        Set<String> uniqueValues = new HashSet<>();
        for (Float pred : predictions.values()) {
            uniqueValues.add(String.format("%.3f", pred));
        }
        
        System.out.printf("\nUnique predictions: %d out of %d\n", 
            uniqueValues.size(), predictions.size());
        
        assertTrue(uniqueValues.size() >= 4, 
            "Should have diverse initial predictions");
        
        // Train specific combinations
        System.out.println("\nTraining specific combinations:");
        Map<String, Object> input00A = new HashMap<>();
        input00A.put("item", 0);
        input00A.put("category", "A");
        
        Map<String, Object> input1B = new HashMap<>();
        input1B.put("item", 1);
        input1B.put("category", "B");
        
        for (int i = 0; i < 50; i++) {
            model.train(input00A, 0.1f);  // Low target
            model.train(input1B, 0.9f);   // High target
        }
        
        System.out.println("\nAfter training:");
        System.out.printf("  Item0_catA: %.3f (trained to 0.1)\n", 
            model.predictFloat(input00A));
        System.out.printf("  Item1_catB: %.3f (trained to 0.9)\n", 
            model.predictFloat(input1B));
        
        // Check untrained combinations
        Map<String, Object> input0B = new HashMap<>();
        input0B.put("item", 0);
        input0B.put("category", "B");
        
        Map<String, Object> input1A = new HashMap<>();
        input1A.put("item", 1);  
        input1A.put("category", "A");
        
        System.out.printf("  Item0_catB: %.3f (untrained)\n", 
            model.predictFloat(input0B));
        System.out.printf("  Item1_catA: %.3f (untrained)\n", 
            model.predictFloat(input1A));
    }
    
    @Test
    public void testEmbeddingOptimizerState() {
        System.out.println("\n=== TESTING EMBEDDING OPTIMIZER STATE ===\n");
        
        Feature[] features = {
            Feature.embedding(2, 4, "item")
        };
        
        // AdamW should maintain momentum for each embedding
        AdamWOptimizer optimizer = new AdamWOptimizer(0.01f, 0.0f);
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(features.length)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(features))
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        Map<String, Object> input0 = new HashMap<>();
        input0.put("item", 0);
        
        // Track predictions over time
        List<Float> predictions = new ArrayList<>();
        
        // Train with consistent target
        for (int i = 0; i < 20; i++) {
            predictions.add(model.predictFloat(input0));
            model.train(input0, 1.0f);
        }
        
        System.out.println("Prediction trajectory (should accelerate due to momentum):");
        for (int i = 0; i < predictions.size(); i++) {
            System.out.printf("  Step %2d: %.3f\n", i, predictions.get(i));
        }
        
        // Check if predictions are moving toward target
        float firstPred = predictions.get(0);
        float lastPred = predictions.get(predictions.size() - 1);
        
        assertTrue(lastPred > firstPred + 0.1f, 
            "Predictions should move significantly toward target");
        
        // Check acceleration (momentum effect)
        float earlyChange = predictions.get(5) - predictions.get(0);
        float lateChange = predictions.get(19) - predictions.get(14);
        
        System.out.printf("\nEarly change (steps 0-5): %.3f\n", earlyChange);
        System.out.printf("Late change (steps 14-19): %.3f\n", lateChange);
        
        // With momentum, later changes might be larger
        // (though this depends on how close we are to target)
    }
}