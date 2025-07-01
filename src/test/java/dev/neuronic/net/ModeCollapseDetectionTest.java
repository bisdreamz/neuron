package dev.neuronic.net;

import dev.neuronic.net.layers.Feature;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import dev.neuronic.net.simple.SimpleNet;
import dev.neuronic.net.simple.SimpleNetFloat;
import org.junit.jupiter.api.Test;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test specifically designed to detect mode collapse in neural networks.
 * This should be added to any neural network test suite to catch convergence issues.
 */
public class ModeCollapseDetectionTest {
    
    @Test
    public void testMixedFeaturesDontCollapse() {
        // This test would have caught the production issue
        Feature[] features = {
            Feature.oneHot(10, "category"),
            Feature.embedding(100, 16, "item_id"),
            Feature.passthrough("numeric_feature")
        };
        
        AdamWOptimizer optimizer = new AdamWOptimizer(0.001f, 0.01f);
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(features.length)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(features))
            .layer(Layers.hiddenDenseRelu(64))
            .layer(Layers.hiddenDenseRelu(32))
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        // Generate diverse training data
        Random rand = new Random(42);
        List<Map<String, Object>> trainingData = new ArrayList<>();
        List<Float> targets = new ArrayList<>();
        
        for (int i = 0; i < 1000; i++) {
            Map<String, Object> input = new HashMap<>();
            input.put("category", rand.nextInt(10));
            input.put("item_id", rand.nextInt(100));
            float numericValue = rand.nextFloat() * 10;
            input.put("numeric_feature", numericValue);
            
            // Target based on features (creating learnable patterns)
            float target = (Integer) input.get("category") * 0.1f + 
                          ((Integer) input.get("item_id") % 10) * 0.05f + 
                          numericValue * 0.02f;
            
            trainingData.add(input);
            targets.add(target);
        }
        
        // Train the model
        for (int epoch = 0; epoch < 10; epoch++) {
            for (int i = 0; i < trainingData.size(); i++) {
                model.train(trainingData.get(i), targets.get(i));
            }
        }
        
        // Check for mode collapse
        Set<String> uniquePredictions = new HashSet<>();
        float minPred = Float.MAX_VALUE;
        float maxPred = Float.MIN_VALUE;
        
        for (int i = 0; i < 200; i++) {
            Map<String, Object> testInput = new HashMap<>();
            testInput.put("category", rand.nextInt(10));
            testInput.put("item_id", rand.nextInt(100));
            testInput.put("numeric_feature", rand.nextFloat() * 10);
            
            float pred = model.predictFloat(testInput);
            uniquePredictions.add(String.format("%.4f", pred));
            minPred = Math.min(minPred, pred);
            maxPred = Math.max(maxPred, pred);
        }
        
        System.out.println("Prediction diversity test:");
        System.out.println("Unique predictions: " + uniquePredictions.size() + "/200");
        System.out.println("Prediction range: " + minPred + " to " + maxPred);
        
        // Assertions that would catch mode collapse
        assertTrue(uniquePredictions.size() > 50, 
            "Model has collapsed! Only " + uniquePredictions.size() + " unique predictions out of 200");
        assertTrue(maxPred - minPred > 0.5f, 
            "Model predictions have no variance! Range is only " + (maxPred - minPred));
    }
    
    @Test
    public void testOnlineTrainingWithNegatives() {
        // Simulates your production scenario
        Feature[] features = {
            Feature.embeddingLRU(1000, 32, "app"),
            Feature.oneHot(20, "country"),
            Feature.passthrough("bid_floor")
        };
        
        AdamWOptimizer optimizer = new AdamWOptimizer(0.01f, 0.001f);
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(features.length)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(features))
            .layer(Layers.hiddenDenseRelu(128))
            .layer(Layers.hiddenDenseRelu(64))
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        Random rand = new Random(42);
        
        // Online training simulation
        int bidCount = 0;
        int noBidCount = 0;
        
        for (int i = 0; i < 10000; i++) {
            Map<String, Object> input = new HashMap<>();
            input.put("app", rand.nextInt(1000));
            input.put("country", rand.nextInt(20));
            input.put("bid_floor", rand.nextFloat() * 0.5f);
            
            // 1.5% bid rate
            if (rand.nextFloat() < 0.015f) {
                float bidValue = 0.5f + rand.nextFloat() * 2.5f;
                model.train(input, bidValue);
                bidCount++;
            } else if (rand.nextFloat() < 0.02f) {
                // 2% of non-bids get negative training
                model.train(input, -0.25f);
                noBidCount++;
            }
            
            // Check for collapse every 1000 iterations
            if (i % 1000 == 0 && i > 0) {
                Set<String> predictions = new HashSet<>();
                for (int j = 0; j < 100; j++) {
                    Map<String, Object> testInput = new HashMap<>();
                    testInput.put("app", rand.nextInt(1000));
                    testInput.put("country", rand.nextInt(20));
                    testInput.put("bid_floor", rand.nextFloat() * 0.5f);
                    
                    float pred = model.predictFloat(testInput);
                    predictions.add(String.format("%.3f", pred));
                }
                
                System.out.printf("Iteration %d: %d unique predictions, %d bids, %d no-bids\n", 
                    i, predictions.size(), bidCount, noBidCount);
                
                // Early warning of collapse
                if (predictions.size() < 10) {
                    fail("Mode collapse detected at iteration " + i + 
                         "! Only " + predictions.size() + " unique predictions");
                }
            }
        }
        
        // Final diversity check
        checkPredictionDiversity(model, 1000);
    }
    
    private void checkPredictionDiversity(SimpleNetFloat model, int samples) {
        Random rand = new Random(123);
        Map<String, Integer> predictionCounts = new HashMap<>();
        float sum = 0;
        float sumSquares = 0;
        
        for (int i = 0; i < samples; i++) {
            Map<String, Object> input = new HashMap<>();
            input.put("app", rand.nextInt(1000));
            input.put("country", rand.nextInt(20));
            input.put("bid_floor", rand.nextFloat() * 0.5f);
            
            float pred = model.predictFloat(input);
            String roundedPred = String.format("%.3f", pred);
            predictionCounts.merge(roundedPred, 1, Integer::sum);
            
            sum += pred;
            sumSquares += pred * pred;
        }
        
        float mean = sum / samples;
        float variance = (sumSquares / samples) - (mean * mean);
        float stdDev = (float) Math.sqrt(variance);
        
        System.out.println("\nDiversity Analysis:");
        System.out.println("Unique predictions: " + predictionCounts.size());
        System.out.println("Mean: " + mean);
        System.out.println("Std Dev: " + stdDev);
        
        // Find most common prediction
        int maxCount = 0;
        String mostCommon = "";
        for (Map.Entry<String, Integer> entry : predictionCounts.entrySet()) {
            if (entry.getValue() > maxCount) {
                maxCount = entry.getValue();
                mostCommon = entry.getKey();
            }
        }
        
        System.out.println("Most common prediction: " + mostCommon + 
                          " (appears " + maxCount + "/" + samples + " times)");
        
        // Strict assertions
        assertTrue(predictionCounts.size() > samples / 10, 
            "Insufficient prediction diversity");
        assertTrue(stdDev > 0.1f, 
            "Predictions have too little variance (std dev = " + stdDev + ")");
        assertTrue(maxCount < samples / 2, 
            "Mode collapse: " + mostCommon + " appears in " + 
            (100.0 * maxCount / samples) + "% of predictions");
    }
}