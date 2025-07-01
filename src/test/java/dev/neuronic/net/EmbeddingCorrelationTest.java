package dev.neuronic.net;

import dev.neuronic.net.layers.Feature;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import dev.neuronic.net.simple.SimpleNet;
import dev.neuronic.net.simple.SimpleNetFloat;
import org.junit.jupiter.api.Test;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test that predictions actually correlate with assigned CPM values.
 * This is critical - if the model just predicts averages, it's useless!
 */
public class EmbeddingCorrelationTest {
    
    @Test
    public void testPredictionCorrelation() {
        System.out.println("=== PREDICTION CORRELATION TEST ===\n");
        System.out.println("Testing if model actually learns segment-specific CPMs...\n");
        
        // Test with proper configuration
        testCorrelation(32, 512, true);
        testCorrelation(32, 512, false);
    }
    
    private void testCorrelation(int embeddingDim, int firstLayerSize, boolean useComplexModel) {
        System.out.printf("\n=== Testing: %d dims, %d neurons, %s ===\n", 
            embeddingDim, firstLayerSize, useComplexModel ? "complex features" : "simple features");
        
        Feature[] features;
        if (useComplexModel) {
            // Your full feature set
            features = new Feature[] {
                Feature.oneHot(50, "FORMAT"),
                Feature.oneHot(50, "PLCMT"),
                Feature.oneHot(50, "DEVTYPE"),
                Feature.oneHot(50, "DEVCON"),
                Feature.oneHot(50, "GEO"),
                Feature.oneHot(50, "PUBID"),
                Feature.oneHot(50, "OS"),
                Feature.embeddingLRU(20_000, embeddingDim, "ZONEID"),
                Feature.embeddingLRU(20_000, embeddingDim, "DOMAIN"),
                Feature.embeddingLRU(20_000, embeddingDim, "PUB"),
                Feature.embeddingLRU(20_000, embeddingDim, "SITEID"),
                Feature.autoScale(0f, 20f, "BIDFLOOR"),
                Feature.autoScale(0f, 600f, "TMAX")
            };
        } else {
            // Simplified to isolate the issue
            features = new Feature[] {
                Feature.embeddingLRU(10_000, embeddingDim, "ZONEID"),
                Feature.embeddingLRU(5_000, embeddingDim, "DOMAIN"),
                Feature.passthrough("BIDFLOOR")
            };
        }
        
        AdamWOptimizer optimizer = new AdamWOptimizer(0.001f, 0.001f);
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(features.length)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(features))
            .layer(Layers.hiddenDenseRelu(firstLayerSize))
            .layer(Layers.hiddenDenseRelu(256))
            .layer(Layers.hiddenDenseRelu(128))
            .layer(Layers.hiddenDenseRelu(64))
            .withGlobalGradientClipping(1.0f)
            .output(Layers.outputHuberRegression(1, optimizer, 3.0f));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        Random rand = new Random(42);
        
        // Create KNOWN mappings
        Map<Integer, Float> zoneTargetCPMs = new HashMap<>();
        Map<Integer, Float> domainModifiers = new HashMap<>();
        
        // Assign specific CPMs to zones
        for (int zoneId = 0; zoneId < 1000; zoneId++) {
            float cpm;
            if (zoneId < 10) {
                cpm = 10.0f + zoneId;  // Premium: $10-19
            } else if (zoneId < 100) {
                cpm = 5.0f + (zoneId / 20.0f);  // Good: $5-9
            } else if (zoneId < 500) {
                cpm = 1.0f + (zoneId / 200.0f);  // Regular: $1-3.5
            } else {
                cpm = -0.25f + (zoneId / 1000.0f);  // Poor: -$0.25-0.5
            }
            zoneTargetCPMs.put(zoneId, cpm);
        }
        
        // Assign modifiers to domains
        for (int domainId = 0; domainId < 1000; domainId++) {
            float modifier = 0.8f + (domainId / 2500.0f);  // 0.8-1.2x
            domainModifiers.put(domainId, modifier);
        }
        
        // Training phase
        System.out.println("Training on known zone->CPM mappings...");
        Map<String, Integer> trainingSamples = new HashMap<>();
        
        for (int step = 0; step < 100_000; step++) {
            int zoneId = rand.nextInt(1000);
            int domainId = rand.nextInt(1000);
            
            Map<String, Object> input = new HashMap<>();
            if (useComplexModel) {
                // Add all features
                input.put("FORMAT", rand.nextInt(10));
                input.put("PLCMT", rand.nextInt(5));
                input.put("DEVTYPE", rand.nextInt(7));
                input.put("DEVCON", rand.nextInt(5));
                input.put("GEO", rand.nextInt(30));
                input.put("PUBID", rand.nextInt(50));
                input.put("OS", rand.nextInt(4));
                input.put("PUB", rand.nextInt(1000));
                input.put("SITEID", rand.nextInt(2000));
                input.put("TMAX", 300f);
            }
            
            input.put("ZONEID", zoneId);
            input.put("DOMAIN", domainId);
            input.put("BIDFLOOR", 0.5f);
            
            // Calculate exact target based on zone and domain
            float zoneCPM = zoneTargetCPMs.get(zoneId);
            float domainMod = domainModifiers.get(domainId);
            float target = zoneCPM * domainMod;
            
            // Track training frequency
            String key = zoneId + "_" + domainId;
            trainingSamples.merge(key, 1, Integer::sum);
            
            // Train with 2% probability
            if (rand.nextFloat() < 0.02f) {
                model.train(input, target);
            }
        }
        
        System.out.printf("Trained on %d unique zone-domain combinations\n", trainingSamples.size());
        
        // Evaluation phase - test EXACT predictions
        System.out.println("\nTesting prediction accuracy and correlation...");
        
        List<Float> actualCPMs = new ArrayList<>();
        List<Float> predictedCPMs = new ArrayList<>();
        Map<Integer, List<Float>> predictionsByZone = new HashMap<>();
        
        // Test 500 specific zone-domain combinations
        for (int test = 0; test < 500; test++) {
            int zoneId = test % 1000;
            int domainId = test % 1000;
            
            Map<String, Object> input = new HashMap<>();
            if (useComplexModel) {
                input.put("FORMAT", 0);
                input.put("PLCMT", 0);
                input.put("DEVTYPE", 0);
                input.put("DEVCON", 0);
                input.put("GEO", 0);
                input.put("PUBID", 0);
                input.put("OS", 0);
                input.put("PUB", 0);
                input.put("SITEID", 0);
                input.put("TMAX", 300f);
            }
            
            input.put("ZONEID", zoneId);
            input.put("DOMAIN", domainId);
            input.put("BIDFLOOR", 0.5f);
            
            float prediction = model.predictFloat(input);
            float expected = zoneTargetCPMs.get(zoneId) * domainModifiers.get(domainId);
            
            actualCPMs.add(expected);
            predictedCPMs.add(prediction);
            
            predictionsByZone.computeIfAbsent(zoneId, k -> new ArrayList<>()).add(prediction);
        }
        
        // Calculate correlation
        double correlation = calculateCorrelation(actualCPMs, predictedCPMs);
        System.out.printf("Correlation between actual and predicted: %.3f\n", correlation);
        
        // Check prediction diversity
        Set<String> uniquePredictions = new HashSet<>();
        for (float pred : predictedCPMs) {
            uniquePredictions.add(String.format("%.3f", pred));
        }
        System.out.printf("Unique predictions: %d out of %d (%.1f%%)\n", 
            uniquePredictions.size(), predictedCPMs.size(), 
            100.0 * uniquePredictions.size() / predictedCPMs.size());
        
        // Test if model differentiates between zones
        System.out.println("\nTesting zone differentiation:");
        testZoneDifferentiation(model, zoneTargetCPMs, useComplexModel);
        
        // Show sample predictions vs actual
        System.out.println("\nSample predictions (first 10):");
        for (int i = 0; i < 10; i++) {
            System.out.printf("  Zone %d: Expected=$%.2f, Predicted=$%.2f, Error=$%.2f\n",
                i, actualCPMs.get(i), predictedCPMs.get(i), 
                Math.abs(actualCPMs.get(i) - predictedCPMs.get(i)));
        }
        
        // Verdict
        if (correlation < 0.5) {
            System.out.println("\n❌ FAILURE: Model is NOT learning zone-specific CPMs!");
            System.out.println("   Low correlation indicates predictions don't match training targets.");
        } else if (uniquePredictions.size() < predictedCPMs.size() * 0.8) {
            System.out.println("\n❌ FAILURE: Model has low prediction diversity!");
            System.out.println("   Many segments are getting the same prediction.");
        } else {
            System.out.println("\n✓ SUCCESS: Model correctly learns zone-specific CPMs!");
        }
    }
    
    private void testZoneDifferentiation(SimpleNetFloat model, Map<Integer, Float> zoneTargetCPMs, 
                                         boolean useComplexModel) {
        // Test if premium zones predict higher than poor zones
        Map<String, Object> baseInput = new HashMap<>();
        if (useComplexModel) {
            baseInput.put("FORMAT", 0);
            baseInput.put("PLCMT", 0);
            baseInput.put("DEVTYPE", 0);
            baseInput.put("DEVCON", 0);
            baseInput.put("GEO", 0);
            baseInput.put("PUBID", 0);
            baseInput.put("OS", 0);
            baseInput.put("PUB", 0);
            baseInput.put("SITEID", 0);
            baseInput.put("TMAX", 300f);
        }
        baseInput.put("DOMAIN", 100);
        baseInput.put("BIDFLOOR", 0.5f);
        
        // Test premium zone (id=5)
        Map<String, Object> premiumInput = new HashMap<>(baseInput);
        premiumInput.put("ZONEID", 5);
        float premiumPred = model.predictFloat(premiumInput);
        
        // Test good zone (id=50)
        Map<String, Object> goodInput = new HashMap<>(baseInput);
        goodInput.put("ZONEID", 50);
        float goodPred = model.predictFloat(goodInput);
        
        // Test regular zone (id=300)
        Map<String, Object> regularInput = new HashMap<>(baseInput);
        regularInput.put("ZONEID", 300);
        float regularPred = model.predictFloat(regularInput);
        
        // Test poor zone (id=800)
        Map<String, Object> poorInput = new HashMap<>(baseInput);
        poorInput.put("ZONEID", 800);
        float poorPred = model.predictFloat(poorInput);
        
        System.out.printf("Premium (zone 5): Expected=$%.2f, Predicted=$%.2f\n", 
            zoneTargetCPMs.get(5), premiumPred);
        System.out.printf("Good (zone 50): Expected=$%.2f, Predicted=$%.2f\n", 
            zoneTargetCPMs.get(50), goodPred);
        System.out.printf("Regular (zone 300): Expected=$%.2f, Predicted=$%.2f\n", 
            zoneTargetCPMs.get(300), regularPred);
        System.out.printf("Poor (zone 800): Expected=$%.2f, Predicted=$%.2f\n", 
            zoneTargetCPMs.get(800), poorPred);
        
        boolean correctOrder = premiumPred > goodPred && goodPred > regularPred && regularPred > poorPred;
        System.out.printf("Correct ordering: %s\n", correctOrder ? "YES" : "NO");
    }
    
    private double calculateCorrelation(List<Float> x, List<Float> y) {
        int n = x.size();
        double sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0, sumY2 = 0;
        
        for (int i = 0; i < n; i++) {
            sumX += x.get(i);
            sumY += y.get(i);
            sumXY += x.get(i) * y.get(i);
            sumX2 += x.get(i) * x.get(i);
            sumY2 += y.get(i) * y.get(i);
        }
        
        double numerator = n * sumXY - sumX * sumY;
        double denominator = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
        
        if (denominator == 0) return 0;
        return numerator / denominator;
    }
}