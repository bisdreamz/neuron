package dev.neuronic.net;

import dev.neuronic.net.layers.Feature;
import dev.neuronic.net.layers.Layer;
import dev.neuronic.net.layers.MixedFeatureInputLayer;
import dev.neuronic.net.optimizers.SgdOptimizer;
import org.junit.jupiter.api.Test;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Simplest possible test to isolate the bug.
 * TODO: REMOVE THIS FILE AFTER DEBUGGING
 */
public class SimplestDebugTest {
    
    @Test
    public void testDirectEmbeddingAccess() {
        System.out.println("=== DIRECT EMBEDDING ACCESS TEST ===\n");
        
        // Direct access to MixedFeatureInputLayer
        Feature[] features = {
            Feature.embedding(3, 2, "item")  // 3 items, 2-dim embeddings
        };
        
        SgdOptimizer optimizer = new SgdOptimizer(1.0f);
        MixedFeatureInputLayer layer = new MixedFeatureInputLayer(optimizer, features, 
            dev.neuronic.net.WeightInitStrategy.XAVIER);
        
        // Check initial embeddings
        System.out.println("Initial embeddings:");
        for (int i = 0; i < 3; i++) {
            float[] emb = layer.getEmbedding(0, i);
            System.out.printf("  Item %d: [%.3f, %.3f]\n", i, emb[0], emb[1]);
        }
        
        // Forward pass with item 0
        float[] input = {0.0f};
        Layer.LayerContext ctx = layer.forward(input, true);
        System.out.println("\nForward output: [" + ctx.outputs()[0] + ", " + ctx.outputs()[1] + "]");
        
        // Backward pass with gradient
        float[] gradient = {1.0f, 1.0f};
        Layer.LayerContext[] stack = {ctx};
        layer.backward(stack, 0, gradient);
        
        // Apply gradients
        layer.applyGradients(null, null);
        
        // Check embeddings after update
        System.out.println("\nEmbeddings after update:");
        for (int i = 0; i < 3; i++) {
            float[] emb = layer.getEmbedding(0, i);
            System.out.printf("  Item %d: [%.3f, %.3f]\n", i, emb[0], emb[1]);
        }
    }
    
    @Test
    public void testLinearRegressionOutput() {
        System.out.println("\n=== LINEAR REGRESSION OUTPUT TEST ===\n");
        
        // Just embedding -> output (no hidden layer)
        Feature[] features = {
            Feature.embedding(3, 1, "item")  // 3 items, 1-dim embeddings
        };
        
        SgdOptimizer optimizer = new SgdOptimizer(0.1f);
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(features.length)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(features))
            .output(Layers.outputLinearRegression(1));
        
        // Get layers
        Layer[] layers = net.getLayers();
        MixedFeatureInputLayer inputLayer = (MixedFeatureInputLayer) layers[0];
        
        // Set specific embeddings for clarity
        System.out.println("Setting embeddings manually:");
        // Can't set embeddings directly, so let's just check what happens
        
        // Forward pass for each item
        System.out.println("\nInitial predictions:");
        for (int i = 0; i < 3; i++) {
            float[] input = {(float)i};
            float pred = net.predict(input)[0];
            System.out.printf("  Item %d: %.3f\n", i, pred);
        }
        
        // Train item 0
        System.out.println("\nTraining item 0 -> 1.0");
        float[][] inputs = {{0.0f}};
        float[][] targets = {{1.0f}};
        net.trainBatch(inputs, targets);
        
        // Check predictions after
        System.out.println("\nPredictions after training:");
        for (int i = 0; i < 3; i++) {
            float[] input = {(float)i};
            float pred = net.predict(input)[0];
            System.out.printf("  Item %d: %.3f\n", i, pred);
        }
        
        // Check embeddings
        System.out.println("\nEmbeddings after training:");
        for (int i = 0; i < 3; i++) {
            float[] emb = inputLayer.getEmbedding(0, i);
            System.out.printf("  Item %d: [%.3f]\n", i, emb[0]);
        }
    }
}