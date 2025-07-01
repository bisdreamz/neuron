package dev.neuronic.net;

import dev.neuronic.net.layers.Feature;
import dev.neuronic.net.layers.MixedFeatureInputLayer;
import dev.neuronic.net.layers.Layer;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import org.junit.jupiter.api.Test;

import java.util.*;

/**
 * Test to understand gradient magnitudes and why embeddings collapse
 */
public class GradientMagnitudeTest {
    
    @Test
    public void testGradientMagnitudes() {
        System.out.println("=== GRADIENT MAGNITUDE TEST ===\n");
        
        // Start with small embedding table
        Feature[] features = {
            Feature.embedding(100, 8, "segment")
        };
        
        AdamWOptimizer optimizer = new AdamWOptimizer(0.01f, 0.0f);
        
        NeuralNet net = NeuralNet.newBuilder()
                .setDefaultOptimizer(optimizer)
                .layer(Layers.inputMixed(features))
                .layer(Layers.hiddenDenseRelu(16))
                .output(Layers.outputLinearRegression(1));
        
        // Hook to capture gradients
        MixedFeatureInputLayer inputLayer = (MixedFeatureInputLayer) net.getInputLayer();
        
        // Train one sample and check gradient magnitude
        System.out.println("Training single sample: segment 0 -> 10.0");
        float[] input = {0};
        float[] target = {10.0f};
        
        // Get initial embedding
        float[] embBefore = inputLayer.getEmbedding(0, 0).clone();
        
        // Train
        net.train(input, target);
        
        // Get updated embedding
        float[] embAfter = inputLayer.getEmbedding(0, 0);
        
        // Calculate change
        float changeNorm = 0;
        for (int i = 0; i < embBefore.length; i++) {
            float diff = embAfter[i] - embBefore[i];
            changeNorm += diff * diff;
        }
        changeNorm = (float)Math.sqrt(changeNorm);
        
        System.out.printf("Embedding change norm: %.6f\n", changeNorm);
        System.out.printf("Before: [%.3f, %.3f, ...]\n", embBefore[0], embBefore[1]);
        System.out.printf("After:  [%.3f, %.3f, ...]\n", embAfter[0], embAfter[1]);
        
        // Now test with large embedding table
        System.out.println("\n--- Testing with large embedding table ---");
        
        Feature[] largeFeatures = {
            Feature.embedding(10000, 32, "segment")
        };
        
        NeuralNet largeNet = NeuralNet.newBuilder()
                .setDefaultOptimizer(optimizer)
                .layer(Layers.inputMixed(largeFeatures))
                .layer(Layers.hiddenDenseRelu(64))
                .output(Layers.outputLinearRegression(1));
        
        MixedFeatureInputLayer largeInputLayer = (MixedFeatureInputLayer) largeNet.getInputLayer();
        
        // Train same pattern
        float[] largeEmbBefore = largeInputLayer.getEmbedding(0, 0).clone();
        largeNet.train(input, target);
        float[] largeEmbAfter = largeInputLayer.getEmbedding(0, 0);
        
        // Calculate change
        float largeChangeNorm = 0;
        for (int i = 0; i < largeEmbBefore.length; i++) {
            float diff = largeEmbAfter[i] - largeEmbBefore[i];
            largeChangeNorm += diff * diff;
        }
        largeChangeNorm = (float)Math.sqrt(largeChangeNorm);
        
        System.out.printf("Large embedding change norm: %.6f\n", largeChangeNorm);
        System.out.printf("Before: [%.3f, %.3f, ...]\n", largeEmbBefore[0], largeEmbBefore[1]);
        System.out.printf("After:  [%.3f, %.3f, ...]\n", largeEmbAfter[0], largeEmbAfter[1]);
        
        // Test predictions
        System.out.println("\n--- Testing prediction spread ---");
        Random rand = new Random(42);
        Set<String> smallPreds = new HashSet<>();
        Set<String> largePreds = new HashSet<>();
        
        for (int i = 0; i < 20; i++) {
            int segmentId = rand.nextInt(50);
            float smallPred = net.predict(new float[]{segmentId})[0];
            float largePred = largeNet.predict(new float[]{segmentId})[0];
            
            smallPreds.add(String.format("%.3f", smallPred));
            largePreds.add(String.format("%.3f", largePred));
        }
        
        System.out.printf("Small network: %d unique predictions\n", smallPreds.size());
        System.out.printf("Large network: %d unique predictions\n", largePreds.size());
        
        if (largePreds.size() <= 2) {
            System.out.println("\n[COLLAPSE DETECTED] Large embedding table leads to collapse!");
        }
    }
}