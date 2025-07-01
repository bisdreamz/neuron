package dev.neuronic.net;

import dev.neuronic.net.layers.Feature;
import dev.neuronic.net.layers.MixedFeatureInputLayer;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import org.junit.jupiter.api.Test;

/**
 * Debug test to understand why embeddings collapse
 */
public class EmbeddingCollapseDebugTest {
    
    @Test
    public void testEmbeddingDifferentiation() {
        System.out.println("=== EMBEDDING DIFFERENTIATION TEST ===\n");
        
        Feature[] features = {
            Feature.embedding(100, 8, "segment")  // Small for easy debugging
        };
        
        AdamWOptimizer optimizer = new AdamWOptimizer(0.1f, 0.0f); // High LR, no decay
        
        NeuralNet net = NeuralNet.newBuilder()
                .setDefaultOptimizer(optimizer)
                .layer(Layers.inputMixed(features))
                .layer(Layers.hiddenDenseRelu(16))
                .output(Layers.outputLinearRegression(1));
        
        // Get the input layer to inspect embeddings
        MixedFeatureInputLayer inputLayer = (MixedFeatureInputLayer) net.getInputLayer();
        
        // Train with very different targets
        System.out.println("Training segment 0 -> 0.0");
        for (int i = 0; i < 100; i++) {
            net.train(new float[]{0}, new float[]{0.0f});
        }
        
        System.out.println("Training segment 1 -> 10.0");
        for (int i = 0; i < 100; i++) {
            net.train(new float[]{1}, new float[]{10.0f});
        }
        
        // Check predictions
        float pred0 = net.predict(new float[]{0})[0];
        float pred1 = net.predict(new float[]{1})[0];
        float pred2 = net.predict(new float[]{2})[0]; // Untrained
        
        System.out.printf("\nPredictions:\n");
        System.out.printf("Segment 0 (trained to 0.0): %.3f\n", pred0);
        System.out.printf("Segment 1 (trained to 10.0): %.3f\n", pred1);
        System.out.printf("Segment 2 (untrained): %.3f\n", pred2);
        System.out.printf("Difference (1-0): %.3f\n", pred1 - pred0);
        
        // Check embedding values
        System.out.println("\nEmbedding values:");
        float[] emb0 = inputLayer.getEmbedding(0, 0);
        float[] emb1 = inputLayer.getEmbedding(0, 1);
        float[] emb2 = inputLayer.getEmbedding(0, 2);
        
        System.out.printf("Embedding 0: [%.3f, %.3f, %.3f, ...]\n", emb0[0], emb0[1], emb0[2]);
        System.out.printf("Embedding 1: [%.3f, %.3f, %.3f, ...]\n", emb1[0], emb1[1], emb1[2]);
        System.out.printf("Embedding 2: [%.3f, %.3f, %.3f, ...]\n", emb2[0], emb2[1], emb2[2]);
        
        // Calculate embedding distances
        float dist01 = euclideanDistance(emb0, emb1);
        float dist02 = euclideanDistance(emb0, emb2);
        float dist12 = euclideanDistance(emb1, emb2);
        
        System.out.printf("\nEmbedding distances:\n");
        System.out.printf("Distance 0-1: %.3f\n", dist01);
        System.out.printf("Distance 0-2: %.3f\n", dist02);
        System.out.printf("Distance 1-2: %.3f\n", dist12);
        
        // Success criteria
        boolean success = Math.abs(pred1 - pred0) > 5.0f;
        System.out.println("\n" + (success ? "[SUCCESS]" : "[FAILURE]") + " - Embeddings " + 
                          (success ? "learned to differentiate" : "failed to differentiate"));
    }
    
    private float euclideanDistance(float[] a, float[] b) {
        float sum = 0;
        for (int i = 0; i < a.length; i++) {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }
        return (float) Math.sqrt(sum);
    }
}