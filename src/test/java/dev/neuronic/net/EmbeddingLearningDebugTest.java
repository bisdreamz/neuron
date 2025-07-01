package dev.neuronic.net;

import dev.neuronic.net.layers.Feature;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import dev.neuronic.net.simple.SimpleNet;
import dev.neuronic.net.simple.SimpleNetFloat;
import org.junit.jupiter.api.Test;

import java.util.*;

/**
 * Debug why embeddings aren't learning properly.
 * Start with the simplest possible case and build up.
 */
public class EmbeddingLearningDebugTest {
    
    @Test
    public void debugEmbeddingLearning() {
        System.out.println("=== EMBEDDING LEARNING DEBUG ===\n");
        
        // Test 1: Can we learn a simple 10-zone mapping?
        testSimpleMapping();
        
        // Test 2: What happens with more zones?
        testScaledMapping(100);
        testScaledMapping(1000);
        
        // Test 3: Effect of training frequency
        testTrainingFrequency();
    }
    
    private void testSimpleMapping() {
        System.out.println("\n=== Test 1: Simple 10-zone mapping ===");
        
        Feature[] features = {
            Feature.embedding(10, 8, "zone")  // Just 10 zones
        };
        
        AdamWOptimizer optimizer = new AdamWOptimizer(0.001f, 0.0f); // No weight decay
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(features.length)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(features))
            .layer(Layers.hiddenDenseRelu(32))
            .withGlobalGradientClipping(1.0f)
            .output(Layers.outputHuberRegression(1, optimizer, 1.0f));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        // Assign specific CPM to each zone
        float[] zoneCPMs = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
        
        // Train each zone many times
        System.out.println("Training 10,000 samples (1,000 per zone)...");
        Random rand = new Random(42);
        
        for (int i = 0; i < 10_000; i++) {
            int zoneId = rand.nextInt(10);
            float target = zoneCPMs[zoneId];
            
            Map<String, Object> input = Map.of("zone", zoneId);
            model.train(input, target);
        }
        
        // Test predictions
        System.out.println("\nPredictions vs Expected:");
        float totalError = 0;
        for (int zone = 0; zone < 10; zone++) {
            float pred = model.predictFloat(Map.of("zone", zone));
            float expected = zoneCPMs[zone];
            float error = Math.abs(pred - expected);
            totalError += error;
            
            System.out.printf("Zone %d: Expected=$%.1f, Predicted=$%.2f, Error=$%.2f %s\n",
                zone, expected, pred, error, error > 0.5f ? "❌" : "✓");
        }
        
        float avgError = totalError / 10;
        System.out.printf("\nAverage error: $%.2f\n", avgError);
        System.out.println(avgError < 0.5f ? "✓ SUCCESS" : "❌ FAILURE");
    }
    
    private void testScaledMapping(int numZones) {
        System.out.printf("\n=== Test 2: Scaled to %d zones ===\n", numZones);
        
        Feature[] features = {
            Feature.embeddingLRU(numZones, 16, "zone")
        };
        
        AdamWOptimizer optimizer = new AdamWOptimizer(0.001f, 0.0f);
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(features.length)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(features))
            .layer(Layers.hiddenDenseRelu(128))
            .layer(Layers.hiddenDenseRelu(64))
            .withGlobalGradientClipping(1.0f)
            .output(Layers.outputHuberRegression(1, optimizer, 1.0f));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        // Create zone->CPM mapping
        Map<Integer, Float> zoneCPMs = new HashMap<>();
        for (int i = 0; i < numZones; i++) {
            zoneCPMs.put(i, 1.0f + (i % 10)); // CPMs from $1-10
        }
        
        // Train
        System.out.printf("Training %d samples...\n", numZones * 100);
        Random rand = new Random(42);
        
        for (int i = 0; i < numZones * 100; i++) {
            int zoneId = rand.nextInt(numZones);
            float target = zoneCPMs.get(zoneId);
            
            Map<String, Object> input = Map.of("zone", zoneId);
            model.train(input, target);
        }
        
        // Test a sample of zones
        System.out.println("\nSample predictions:");
        float totalError = 0;
        int[] testZones = {0, numZones/4, numZones/2, 3*numZones/4, numZones-1};
        
        for (int zone : testZones) {
            float pred = model.predictFloat(Map.of("zone", zone));
            float expected = zoneCPMs.get(zone);
            float error = Math.abs(pred - expected);
            totalError += error;
            
            System.out.printf("Zone %d: Expected=$%.1f, Predicted=$%.2f, Error=$%.2f\n",
                zone, expected, pred, error);
        }
        
        // Test uniqueness
        Set<String> uniquePreds = new HashSet<>();
        for (int i = 0; i < Math.min(100, numZones); i++) {
            float pred = model.predictFloat(Map.of("zone", i));
            uniquePreds.add(String.format("%.3f", pred));
        }
        
        System.out.printf("Unique predictions: %d out of %d\n", 
            uniquePreds.size(), Math.min(100, numZones));
    }
    
    private void testTrainingFrequency() {
        System.out.println("\n=== Test 3: Effect of training frequency ===");
        
        Feature[] features = {
            Feature.embeddingLRU(1000, 32, "zone"),
            Feature.passthrough("bid_floor")
        };
        
        AdamWOptimizer optimizer = new AdamWOptimizer(0.001f, 0.0f);
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(features.length)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(features))
            .layer(Layers.hiddenDenseRelu(256))
            .layer(Layers.hiddenDenseRelu(128))
            .withGlobalGradientClipping(1.0f)
            .output(Layers.outputHuberRegression(1, optimizer, 3.0f));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        // Some zones are seen frequently, others rarely
        Map<Integer, Float> zoneCPMs = new HashMap<>();
        Map<Integer, Integer> zoneFrequency = new HashMap<>();
        
        for (int i = 0; i < 1000; i++) {
            if (i < 10) {
                zoneCPMs.put(i, 5.0f); // Premium zones
                zoneFrequency.put(i, 1000); // Seen frequently
            } else if (i < 100) {
                zoneCPMs.put(i, 2.0f); // Regular zones
                zoneFrequency.put(i, 100); // Seen moderately
            } else {
                zoneCPMs.put(i, 0.5f); // Poor zones
                zoneFrequency.put(i, 10); // Seen rarely
            }
        }
        
        // Train with realistic frequency distribution
        System.out.println("Training with frequency distribution...");
        Random rand = new Random(42);
        int totalSamples = 0;
        
        for (Map.Entry<Integer, Integer> entry : zoneFrequency.entrySet()) {
            int zoneId = entry.getKey();
            int frequency = entry.getValue();
            float target = zoneCPMs.get(zoneId);
            
            for (int i = 0; i < frequency; i++) {
                Map<String, Object> input = new HashMap<>();
                input.put("zone", zoneId);
                input.put("bid_floor", 0.5f + rand.nextFloat());
                
                model.train(input, target);
                totalSamples++;
            }
        }
        
        System.out.printf("Trained on %d total samples\n", totalSamples);
        
        // Test predictions by frequency tier
        System.out.println("\nPredictions by frequency tier:");
        
        // Frequent zones (0-9)
        float freqSum = 0;
        for (int i = 0; i < 10; i++) {
            float pred = model.predictFloat(Map.of("zone", i, "bid_floor", 1.0f));
            freqSum += pred;
        }
        System.out.printf("Frequent zones (1000x training): avg prediction=$%.2f (expected=$5.00)\n", 
            freqSum / 10);
        
        // Moderate zones (10-99)
        float modSum = 0;
        for (int i = 10; i < 100; i++) {
            float pred = model.predictFloat(Map.of("zone", i, "bid_floor", 1.0f));
            modSum += pred;
        }
        System.out.printf("Moderate zones (100x training): avg prediction=$%.2f (expected=$2.00)\n", 
            modSum / 90);
        
        // Rare zones (100-199)
        float rareSum = 0;
        for (int i = 100; i < 200; i++) {
            float pred = model.predictFloat(Map.of("zone", i, "bid_floor", 1.0f));
            rareSum += pred;
        }
        System.out.printf("Rare zones (10x training): avg prediction=$%.2f (expected=$0.50)\n", 
            rareSum / 100);
        
        // Never seen zones (900-999)
        float unseenSum = 0;
        for (int i = 900; i < 1000; i++) {
            float pred = model.predictFloat(Map.of("zone", i, "bid_floor", 1.0f));
            unseenSum += pred;
        }
        System.out.printf("Never seen zones (0x training): avg prediction=$%.2f\n", 
            unseenSum / 100);
    }
}