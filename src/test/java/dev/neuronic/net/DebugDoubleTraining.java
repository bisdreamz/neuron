package dev.neuronic.net;

import dev.neuronic.net.layers.Feature;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import java.util.*;

/**
 * Test if double training (penalty then bid) causes collapse
 */
public class DebugDoubleTraining {
    
    public static void main(String[] args) {
        System.out.println("=== DEBUG DOUBLE TRAINING ===\n");
        
        // Test 1: Single training per input
        testTrainingPattern("Single training", false);
        
        // Test 2: Double training (penalty then bid)
        testTrainingPattern("Double training (penalty then bid)", true);
    }
    
    private static void testTrainingPattern(String name, boolean doubleTrain) {
        System.out.println("\n--- " + name + " ---");
        
        Feature[] features = {
            Feature.embedding(1000, 16, "segment")
        };
        
        AdamWOptimizer optimizer = new AdamWOptimizer(0.001f, 0.01f);
        
        NeuralNet net = NeuralNet.newBuilder()
                .setDefaultOptimizer(optimizer)
                .layer(Layers.inputMixed(features))
                .layer(Layers.hiddenDenseRelu(64))
                .layer(Layers.hiddenDenseRelu(32))
                .output(Layers.outputLinearRegression(1));
        
        Random rand = new Random(42);
        
        // Define segments
        Map<Integer, Float> segmentValues = new HashMap<>();
        for (int i = 0; i < 20; i++) {
            if (i < 5) {
                segmentValues.put(i, 3.0f + i * 0.5f); // Premium: 3.0, 3.5, 4.0, 4.5, 5.0
            } else {
                segmentValues.put(i, 0.5f + (i % 5) * 0.1f); // Regular: 0.5-0.9
            }
        }
        
        // Train
        for (int epoch = 0; epoch < 100; epoch++) {
            for (Map.Entry<Integer, Float> entry : segmentValues.entrySet()) {
                int segmentId = entry.getKey();
                float targetValue = entry.getValue();
                
                if (doubleTrain) {
                    // First train with penalty
                    net.train(new float[]{segmentId}, new float[]{-0.0003f});
                    // Then train with actual value
                    net.train(new float[]{segmentId}, new float[]{targetValue});
                } else {
                    // Just train with actual value
                    net.train(new float[]{segmentId}, new float[]{targetValue});
                }
            }
        }
        
        // Test predictions
        Set<String> uniquePreds = new HashSet<>();
        System.out.println("Predictions:");
        for (int i = 0; i < 10; i++) {
            float pred = net.predict(new float[]{i})[0];
            float expected = segmentValues.get(i);
            uniquePreds.add(String.format("%.3f", pred));
            System.out.printf("  Segment %d: expected=%.1f, predicted=%.3f\n", i, expected, pred);
        }
        
        System.out.printf("Unique predictions: %d/10\n", uniquePreds.size());
        
        if (uniquePreds.size() <= 2) {
            System.out.println("*** COLLAPSE DETECTED! ***");
        }
    }
}