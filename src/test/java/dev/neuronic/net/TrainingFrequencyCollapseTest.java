package dev.neuronic.net;

import dev.neuronic.net.layers.Feature;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import dev.neuronic.net.simple.SimpleNet;
import dev.neuronic.net.simple.SimpleNetFloat;
import org.junit.jupiter.api.Test;

import java.util.*;

/**
 * Test if low training frequency (2% rate) is causing mode collapse.
 * TODO: REMOVE THIS FILE AFTER DEBUGGING
 */
public class TrainingFrequencyCollapseTest {
    
    @Test
    public void testDifferentTrainingFrequencies() {
        System.out.println("=== TRAINING FREQUENCY COLLAPSE TEST ===\n");
        
        // Test different training frequencies
        testTrainingFrequency("FREQ_2_PERCENT", 0.02f);   // Current failing case
        testTrainingFrequency("FREQ_10_PERCENT", 0.10f);  // Higher frequency
        testTrainingFrequency("FREQ_50_PERCENT", 0.50f);  // Much higher
        testTrainingFrequency("FREQ_100_PERCENT", 1.00f); // Train on every sample
    }
    
    private void testTrainingFrequency(String name, float trainFreq) {
        System.out.println("--- " + name + " (train " + (int)(trainFreq * 100) + "% of samples) ---");
        
        // Simple embedding setup to isolate the frequency issue
        Feature[] features = {Feature.embedding(1000, 16, "item")};
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(features.length)
            .setDefaultOptimizer(new AdamWOptimizer(0.01f, 0.001f))
            .layer(Layers.inputMixed(features))
            .layer(Layers.hiddenDenseRelu(64))
            .layer(Layers.hiddenDenseRelu(32))
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        Random rand = new Random(42);
        int totalSamples = 0;
        int trainedSamples = 0;
        
        // Process same number of total samples, but vary training frequency
        for (int step = 0; step < 2000; step++) {
            boolean isGood = rand.nextBoolean();
            
            Map<String, Object> input = new HashMap<>();
            if (isGood) {
                input.put("item", "good_" + rand.nextInt(50));
            } else {
                input.put("item", "bad_" + rand.nextInt(500));
            }
            
            totalSamples++;
            
            // Apply the training frequency
            if (rand.nextFloat() < trainFreq) {
                float target = isGood ? 1.0f : -0.5f;
                model.train(input, target);
                trainedSamples++;
            }
        }
        
        System.out.printf("Total samples: %d, Trained samples: %d (%.1f%%)\n", 
            totalSamples, trainedSamples, 100.0f * trainedSamples / totalSamples);
        
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
        
        System.out.printf("Unique predictions: %d\n", uniquePreds.size());
        System.out.printf("Good avg: %.3f, Bad avg: %.3f\n", goodAvg, badAvg);
        System.out.printf("Discrimination: %.3f\n", goodAvg - badAvg);
        System.out.printf("Result: %s\n\n", collapsed ? "⚠️ COLLAPSED" : "✓ LEARNING");
    }
    
    @Test 
    public void testContinuousVsBatchTraining() {
        System.out.println("=== CONTINUOUS VS BATCH TRAINING ===\n");
        
        // Test 1: Train on every sample as it comes (continuous)
        testContinuousTraining();
        
        // Test 2: Accumulate samples then train in batches
        testBatchAccumulation();
    }
    
    private void testContinuousTraining() {
        System.out.println("--- CONTINUOUS TRAINING (every sample) ---");
        
        Feature[] features = {Feature.embedding(500, 16, "item")};
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(features.length)
            .setDefaultOptimizer(new AdamWOptimizer(0.01f, 0.001f))
            .layer(Layers.inputMixed(features))
            .layer(Layers.hiddenDenseRelu(32))
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        Random rand = new Random(42);
        
        // Train on every single sample
        for (int step = 0; step < 1000; step++) {
            boolean isGood = rand.nextBoolean();
            
            Map<String, Object> input = Map.of("item", 
                isGood ? "good_" + rand.nextInt(20) : "bad_" + rand.nextInt(80));
            float target = isGood ? 1.0f : 0.0f;
            
            model.train(input, target);
        }
        
        testModelDiscrimination(model, "CONTINUOUS");
    }
    
    private void testBatchAccumulation() {
        System.out.println("--- BATCH ACCUMULATION (8 samples per batch) ---");
        
        Feature[] features = {Feature.embedding(500, 16, "item")};
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(features.length)
            .setDefaultOptimizer(new AdamWOptimizer(0.01f, 0.001f))
            .layer(Layers.inputMixed(features))
            .layer(Layers.hiddenDenseRelu(32))
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        Random rand = new Random(42);
        List<Map<String, Object>> batchInputs = new ArrayList<>();
        List<Float> batchTargets = new ArrayList<>();
        
        // Accumulate samples into batches
        for (int step = 0; step < 1000; step++) {
            boolean isGood = rand.nextBoolean();
            
            Map<String, Object> input = Map.of("item", 
                isGood ? "good_" + rand.nextInt(20) : "bad_" + rand.nextInt(80));
            float target = isGood ? 1.0f : 0.0f;
            
            batchInputs.add(input);
            batchTargets.add(target);
            
            // Train when batch is full
            if (batchInputs.size() == 8) {
                for (int i = 0; i < batchInputs.size(); i++) {
                    model.train(batchInputs.get(i), batchTargets.get(i));
                }
                batchInputs.clear();
                batchTargets.clear();
            }
        }
        
        testModelDiscrimination(model, "BATCH");
    }
    
    private void testModelDiscrimination(SimpleNetFloat model, String testName) {
        Set<String> uniquePreds = new HashSet<>();
        float goodSum = 0, badSum = 0;
        
        for (int i = 0; i < 50; i++) {
            Map<String, Object> input = Map.of("item", i < 25 ? "good_0" : "bad_0");
            float pred = model.predictFloat(input);
            uniquePreds.add(String.format("%.3f", pred));
            
            if (i < 25) goodSum += pred;
            else badSum += pred;
        }
        
        float goodAvg = goodSum / 25;
        float badAvg = badSum / 25;
        boolean collapsed = uniquePreds.size() < 5;
        
        System.out.printf("Unique predictions: %d\n", uniquePreds.size());
        System.out.printf("Good avg: %.3f, Bad avg: %.3f\n", goodAvg, badAvg);
        System.out.printf("Discrimination: %.3f\n", goodAvg - badAvg);
        System.out.printf("Result: %s\n\n", collapsed ? "⚠️ COLLAPSED" : "✓ LEARNING");
    }
}