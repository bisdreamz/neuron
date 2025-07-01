package dev.neuronic.net;

import dev.neuronic.net.layers.Feature;
import dev.neuronic.net.layers.MixedFeatureInputLayer;
import dev.neuronic.net.optimizers.AdamWOptimizer;

import java.util.*;

/**
 * Debug why embeddings collapse - simple main method
 */
public class DebugCollapse {
    
    public static void main(String[] args) {
        System.out.println("=== DEBUG EMBEDDING COLLAPSE ===\n");
        
        // Test 1: Small embedding table
        testEmbeddingSize(100, 8, "Small (100x8)");
        
        // Test 2: Large embedding table  
        testEmbeddingSize(10000, 32, "Large (10000x32)");
    }
    
    private static void testEmbeddingSize(int vocabSize, int embDim, String name) {
        System.out.println("\n--- Testing " + name + " ---");
        
        Feature[] features = {
            Feature.embedding(vocabSize, embDim, "segment")
        };
        
        AdamWOptimizer optimizer = new AdamWOptimizer(0.01f, 0.0f);
        
        NeuralNet net = NeuralNet.newBuilder()
                .setDefaultOptimizer(optimizer)
                .layer(Layers.inputMixed(features))
                .layer(Layers.hiddenDenseRelu(64))
                .layer(Layers.hiddenDenseRelu(32))
                .output(Layers.outputLinearRegression(1));
        
        MixedFeatureInputLayer inputLayer = (MixedFeatureInputLayer) net.getInputLayer();
        
        Random rand = new Random(42);
        
        // Train 20 different segments with different values
        for (int i = 0; i < 20; i++) {
            float target = i * 0.5f;  // 0, 0.5, 1.0, ..., 9.5
            
            // Train each segment 10 times
            for (int j = 0; j < 10; j++) {
                net.train(new float[]{i}, new float[]{target});
            }
        }
        
        // Check predictions
        Set<String> uniquePreds = new HashSet<>();
        float minPred = Float.MAX_VALUE;
        float maxPred = Float.MIN_VALUE;
        
        System.out.println("Predictions:");
        for (int i = 0; i < 20; i++) {
            float pred = net.predict(new float[]{i})[0];
            uniquePreds.add(String.format("%.3f", pred));
            minPred = Math.min(minPred, pred);
            maxPred = Math.max(maxPred, pred);
            
            if (i < 5) {
                System.out.printf("  Segment %d (trained to %.1f): %.3f\n", i, i * 0.5f, pred);
            }
        }
        
        System.out.printf("Unique predictions: %d/20\n", uniquePreds.size());
        System.out.printf("Range: [%.3f, %.3f]\n", minPred, maxPred);
        
        // Check untrained segments
        float untrained1 = net.predict(new float[]{50})[0];
        float untrained2 = net.predict(new float[]{vocabSize - 1})[0];
        System.out.printf("Untrained segments: %.3f, %.3f\n", untrained1, untrained2);
        
        // Check embedding norms
        float avgNorm = 0;
        for (int i = 0; i < 5; i++) {
            float[] emb = inputLayer.getEmbedding(0, i);
            float norm = 0;
            for (float v : emb) norm += v * v;
            norm = (float)Math.sqrt(norm);
            avgNorm += norm;
        }
        avgNorm /= 5;
        System.out.printf("Average embedding norm (first 5): %.3f\n", avgNorm);
        
        if (uniquePreds.size() <= 2) {
            System.out.println("*** COLLAPSE DETECTED! ***");
        } else {
            System.out.println("Network learned successfully");
        }
    }
}