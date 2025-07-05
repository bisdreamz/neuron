package dev.neuronic.net;

import dev.neuronic.net.layers.Feature;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import dev.neuronic.net.simple.SimpleNet;
import dev.neuronic.net.simple.SimpleNetFloat;
import org.junit.jupiter.api.Test;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test to verify dropout behavior during inference.
 */
public class DropoutInferenceTest {
    
    @Test
    public void testDropoutDeterministicDuringInference() {
        // Create a simple network with dropout
        Feature[] features = {
            Feature.passthrough("x"),
            Feature.passthrough("y")
        };
        
        AdamWOptimizer optimizer = new AdamWOptimizer(0.01f, 0.0f);
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(features.length)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(features))
            .layer(Layers.hiddenDenseRelu(64))
            .layer(Layers.dropout(0.5f))  // 50% dropout - should be very noticeable
            .layer(Layers.hiddenDenseRelu(32))
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        // Train the model briefly
        Random rand = new Random(42);
        for (int i = 0; i < 100; i++) {
            Map<String, Object> input = new HashMap<>();
            input.put("x", rand.nextFloat());
            input.put("y", rand.nextFloat());
            float target = (Float) input.get("x") + (Float) input.get("y");
            model.train(input, target);
        }
        
        // Test inference - predictions should be deterministic
        Map<String, Object> testInput = new HashMap<>();
        testInput.put("x", 0.5f);
        testInput.put("y", 0.3f);
        
        // Make 10 predictions with same input
        float[] predictions = new float[10];
        for (int i = 0; i < 10; i++) {
            predictions[i] = model.predictFloat(testInput);
        }
        
        // Check if predictions are identical (deterministic)
        float firstPred = predictions[0];
        boolean allSame = true;
        for (int i = 1; i < predictions.length; i++) {
            if (Math.abs(predictions[i] - firstPred) > 1e-6f) {
                allSame = false;
                break;
            }
        }
        
        System.out.println("Predictions with dropout:");
        for (int i = 0; i < predictions.length; i++) {
            System.out.printf("  Prediction %d: %.6f\n", i + 1, predictions[i]);
        }
        
        if (!allSame) {
            // Calculate variance
            float mean = 0;
            for (float p : predictions) mean += p;
            mean /= predictions.length;
            
            float variance = 0;
            for (float p : predictions) {
                float diff = p - mean;
                variance += diff * diff;
            }
            variance /= predictions.length;
            
            System.out.printf("\nVariance in predictions: %.6f\n", variance);
            System.out.println("CRITICAL BUG: Dropout is active during inference!");
        }
        
        assertTrue(allSame, 
            "Predictions should be deterministic during inference, but dropout is causing random variations");
    }
    
    @Test
    public void testDropoutProperlyDisabledDuringInference() {
        System.out.println("\n=== Testing Dropout Properly Disabled During Inference ===\n");
        
        // Train two identical networks - one with dropout, one without
        Feature[] features = {
            Feature.embedding(100, 16, "item"),
            Feature.oneHot(10, "category")
        };
        
        AdamWOptimizer optimizer = new AdamWOptimizer(0.01f, 0.001f);
        
        // Network WITH dropout
        NeuralNet netWithDropout = NeuralNet.newBuilder()
            .input(features.length)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(features))
            .layer(Layers.hiddenDenseRelu(64))
            .layer(Layers.dropout(0.5f))  // High dropout
            .layer(Layers.hiddenDenseRelu(32))
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat modelWithDropout = SimpleNet.ofFloatRegression(netWithDropout);
        
        // Network WITHOUT dropout
        NeuralNet netWithoutDropout = NeuralNet.newBuilder()
            .input(features.length)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(features))
            .layer(Layers.hiddenDenseRelu(64))
            .layer(Layers.hiddenDenseRelu(32))
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat modelWithoutDropout = SimpleNet.ofFloatRegression(netWithoutDropout);
        
        // Train both models with same data
        Random rand = new Random(42);
        List<Map<String, Object>> trainingData = new ArrayList<>();
        List<Float> targets = new ArrayList<>();
        
        for (int i = 0; i < 500; i++) {
            Map<String, Object> input = new HashMap<>();
            input.put("item", rand.nextInt(100));
            input.put("category", rand.nextInt(10));
            
            float target = (Integer) input.get("item") * 0.01f + 
                          (Integer) input.get("category") * 0.1f;
            
            trainingData.add(input);
            targets.add(target);
        }
        
        // Train both models
        for (int epoch = 0; epoch < 10; epoch++) {
            for (int i = 0; i < trainingData.size(); i++) {
                modelWithDropout.train(trainingData.get(i), targets.get(i));
                modelWithoutDropout.train(trainingData.get(i), targets.get(i));
            }
        }
        
        // Analyze prediction diversity
        System.out.println("Model WITH dropout (properly disabled during inference):");
        PredictionStats withDropout = analyzePredictions(modelWithDropout, 200);
        
        System.out.println("\nModel WITHOUT dropout:");
        PredictionStats withoutDropout = analyzePredictions(modelWithoutDropout, 200);
        
        System.out.println("\nComparison:");
        System.out.printf("  Diversity WITH dropout: %d unique values\n", withDropout.uniqueCount);
        System.out.printf("  Diversity WITHOUT dropout: %d unique values\n", withoutDropout.uniqueCount);
        System.out.printf("  Std Dev WITH dropout: %.4f\n", withDropout.stdDev);
        System.out.printf("  Std Dev WITHOUT dropout: %.4f\n", withoutDropout.stdDev);
        
        // Both models should have similar diversity since dropout is disabled during inference
        // Allow for some variance (within 20% of each other)
        float ratio = (float) withDropout.uniqueCount / withoutDropout.uniqueCount;
        assertTrue(ratio > 0.8f && ratio < 1.2f,
            "Models should have similar diversity when dropout is disabled during inference. " +
            "Got ratio: " + ratio);
    }
    
    private PredictionStats analyzePredictions(SimpleNetFloat model, int samples) {
        Random rand = new Random(123);
        Set<String> uniquePreds = new HashSet<>();
        float sum = 0, sumSquares = 0;
        float min = Float.MAX_VALUE, max = Float.MIN_VALUE;
        
        for (int i = 0; i < samples; i++) {
            Map<String, Object> input = new HashMap<>();
            input.put("item", rand.nextInt(100));
            input.put("category", rand.nextInt(10));
            
            float pred = model.predictFloat(input);
            uniquePreds.add(String.format("%.4f", pred));
            
            sum += pred;
            sumSquares += pred * pred;
            min = Math.min(min, pred);
            max = Math.max(max, pred);
        }
        
        float mean = sum / samples;
        float variance = (sumSquares / samples) - (mean * mean);
        float stdDev = (float) Math.sqrt(Math.max(0, variance));
        
        System.out.printf("  Unique predictions: %d/%d\n", uniquePreds.size(), samples);
        System.out.printf("  Range: [%.4f, %.4f]\n", min, max);
        System.out.printf("  Mean: %.4f, Std Dev: %.4f\n", mean, stdDev);
        
        return new PredictionStats(uniquePreds.size(), mean, stdDev, min, max);
    }
    
    private static class PredictionStats {
        final int uniqueCount;
        final float mean, stdDev, min, max;
        
        PredictionStats(int uniqueCount, float mean, float stdDev, float min, float max) {
            this.uniqueCount = uniqueCount;
            this.mean = mean;
            this.stdDev = stdDev;
            this.min = min;
            this.max = max;
        }
    }
}