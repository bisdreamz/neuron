package dev.neuronic.net;

import dev.neuronic.net.layers.Feature;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import dev.neuronic.net.simple.SimpleNet;
import dev.neuronic.net.simple.SimpleNetFloat;
import org.junit.jupiter.api.Test;

import java.util.*;

/**
 * Test the impact of embedding dimensions (16 vs 32 vs 64) on CPM prediction accuracy
 * with 10,000 unique zone IDs and 5,000 unique app bundles.
 */
public class EmbeddingDimensionImpactTest {
    
    @Test
    public void testEmbeddingDimensionImpact() {
        System.out.println("=== EMBEDDING DIMENSION IMPACT TEST ===\n");
        System.out.println("Testing with 10,000 zones and 5,000 app bundles\n");
        
        // Test different embedding dimensions
        testConfiguration(16, 256, "16 dims, 256 neurons");
        testConfiguration(32, 256, "32 dims, 256 neurons");
        testConfiguration(64, 256, "64 dims, 256 neurons");
        testConfiguration(32, 512, "32 dims, 512 neurons (your proposal)");
        testConfiguration(32, 1024, "32 dims, 1024 neurons (extra large)");
    }
    
    private void testConfiguration(int embeddingDim, int firstLayerSize, String configName) {
        System.out.printf("\n=== Testing: %s ===\n", configName);
        
        // Your actual features with variable embedding dimensions
        Feature[] features = {
            Feature.oneHot(50, "FORMAT"),
            Feature.oneHot(50, "PLCMT"),
            Feature.oneHot(50, "DEVTYPE"),
            Feature.oneHot(50, "DEVCON"),
            Feature.oneHot(50, "GEO"),
            Feature.oneHot(50, "PUBID"),
            Feature.oneHot(50, "OS"),
            Feature.oneHot(50, "COMPANYID"),
            Feature.oneHot(50, "SIZE"),
            Feature.oneHot(50, "VDUR"),
            Feature.oneHot(50, "GEOTYPE"),
            Feature.embeddingLRU(20_000, embeddingDim, "PUB"),
            Feature.embeddingLRU(20_000, embeddingDim, "SITEID"),
            Feature.embeddingLRU(20_000, embeddingDim, "DOMAIN"),
            Feature.embeddingLRU(20_000, embeddingDim, "ZONEID"),
            Feature.autoScale(0f, 20f, "BIDFLOOR"),
            Feature.autoScale(0f, 600f, "TMAX")
        };
        
        // Calculate parameter counts
        int oneHotParams = 11 * 50;  // 11 oneHot features
        int embeddingParams = 4 * 20_000 * embeddingDim;  // 4 embedding features
        int totalInputParams = oneHotParams + embeddingParams;
        
        System.out.printf("Input parameters: %,d (%.1f MB)\n", 
            totalInputParams, totalInputParams * 4.0 / 1024 / 1024);
        
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
        
        // Create realistic data distribution
        Random rand = new Random(42);
        
        // Define 10,000 zones with known CPM patterns
        Map<Integer, Float> zoneCPMs = new HashMap<>();
        for (int i = 0; i < 10_000; i++) {
            float baseCPM;
            if (i < 100) {
                baseCPM = 4.0f + rand.nextFloat() * 2.0f;  // Premium zones: $4-6
            } else if (i < 1000) {
                baseCPM = 2.0f + rand.nextFloat() * 2.0f;  // Good zones: $2-4
            } else if (i < 5000) {
                baseCPM = 0.5f + rand.nextFloat() * 1.5f;  // Regular zones: $0.5-2
            } else {
                baseCPM = -0.25f + rand.nextFloat() * 0.5f; // Poor zones: -$0.25-0.25
            }
            zoneCPMs.put(i, baseCPM);
        }
        
        // Define 5,000 domains with quality modifiers
        Map<Integer, Float> domainModifiers = new HashMap<>();
        for (int i = 0; i < 5_000; i++) {
            float modifier;
            if (i < 50) {
                modifier = 1.5f;  // Premium domains
            } else if (i < 500) {
                modifier = 1.2f;  // Good domains
            } else if (i < 2000) {
                modifier = 1.0f;  // Average domains
            } else {
                modifier = 0.8f;  // Below average domains
            }
            domainModifiers.put(i, modifier);
        }
        
        // Training phase
        System.out.println("Training on 50,000 samples...");
        long startTime = System.currentTimeMillis();
        
        for (int step = 0; step < 50_000; step++) {
            Map<String, Object> input = new HashMap<>();
            
            // Random features
            input.put("FORMAT", rand.nextInt(10));
            input.put("PLCMT", rand.nextInt(5));
            input.put("DEVTYPE", rand.nextInt(7));
            input.put("DEVCON", rand.nextInt(5));
            input.put("GEO", rand.nextInt(30));
            input.put("PUBID", rand.nextInt(50));
            input.put("OS", rand.nextInt(4));
            input.put("COMPANYID", rand.nextInt(20));
            input.put("SIZE", rand.nextInt(10));
            input.put("VDUR", rand.nextInt(5));
            input.put("GEOTYPE", rand.nextInt(3));
            
            // Key features that determine CPM
            int zoneId = rand.nextInt(10_000);
            int domainId = rand.nextInt(5_000);
            int pubId = rand.nextInt(1_000);
            int siteId = rand.nextInt(2_000);
            
            input.put("ZONEID", zoneId);
            input.put("DOMAIN", domainId);
            input.put("PUB", pubId);
            input.put("SITEID", siteId);
            input.put("BIDFLOOR", 0.1f + rand.nextFloat() * 2.0f);
            input.put("TMAX", 100f + rand.nextFloat() * 400f);
            
            // Calculate target CPM based on zone and domain quality
            float baseCPM = zoneCPMs.get(zoneId);
            float domainMod = domainModifiers.get(domainId);
            float target = baseCPM * domainMod + (rand.nextFloat() - 0.5f) * 0.5f; // Add noise
            
            // Train with 2% probability
            if (rand.nextFloat() < 0.02f) {
                model.train(input, target);
            }
            
            if (step > 0 && step % 10000 == 0) {
                System.out.printf("  Step %d completed\n", step);
            }
        }
        
        long trainingTime = System.currentTimeMillis() - startTime;
        System.out.printf("Training time: %.1f seconds\n", trainingTime / 1000.0);
        
        // Evaluation phase
        System.out.println("\nEvaluating on test segments...");
        
        // Test on known premium, good, regular, and poor zones
        float[] errors = new float[4];
        int[] counts = new int[4];
        List<Float> allPredictions = new ArrayList<>();
        List<Float> allTargets = new ArrayList<>();
        
        for (int test = 0; test < 1000; test++) {
            // Test different zone quality tiers
            int tier = test % 4;
            int zoneId;
            String tierName;
            
            switch (tier) {
                case 0: // Premium
                    zoneId = rand.nextInt(100);
                    tierName = "Premium";
                    break;
                case 1: // Good
                    zoneId = 100 + rand.nextInt(900);
                    tierName = "Good";
                    break;
                case 2: // Regular
                    zoneId = 1000 + rand.nextInt(4000);
                    tierName = "Regular";
                    break;
                default: // Poor
                    zoneId = 5000 + rand.nextInt(5000);
                    tierName = "Poor";
                    break;
            }
            
            Map<String, Object> input = new HashMap<>();
            input.put("FORMAT", rand.nextInt(10));
            input.put("PLCMT", rand.nextInt(5));
            input.put("DEVTYPE", rand.nextInt(7));
            input.put("DEVCON", rand.nextInt(5));
            input.put("GEO", rand.nextInt(30));
            input.put("PUBID", rand.nextInt(50));
            input.put("OS", rand.nextInt(4));
            input.put("COMPANYID", rand.nextInt(20));
            input.put("SIZE", rand.nextInt(10));
            input.put("VDUR", rand.nextInt(5));
            input.put("GEOTYPE", rand.nextInt(3));
            
            int domainId = rand.nextInt(5_000);
            input.put("ZONEID", zoneId);
            input.put("DOMAIN", domainId);
            input.put("PUB", rand.nextInt(1_000));
            input.put("SITEID", rand.nextInt(2_000));
            input.put("BIDFLOOR", 0.5f);
            input.put("TMAX", 300f);
            
            float prediction = model.predictFloat(input);
            float expectedCPM = zoneCPMs.get(zoneId) * domainModifiers.get(domainId);
            float error = Math.abs(prediction - expectedCPM);
            
            errors[tier] += error;
            counts[tier]++;
            allPredictions.add(prediction);
            allTargets.add(expectedCPM);
        }
        
        // Calculate metrics
        System.out.println("\nResults by tier:");
        System.out.printf("  Premium zones: avg error=$%.3f\n", errors[0] / counts[0]);
        System.out.printf("  Good zones: avg error=$%.3f\n", errors[1] / counts[1]);
        System.out.printf("  Regular zones: avg error=$%.3f\n", errors[2] / counts[2]);
        System.out.printf("  Poor zones: avg error=$%.3f\n", errors[3] / counts[3]);
        
        float totalError = (errors[0] + errors[1] + errors[2] + errors[3]) / 1000;
        System.out.printf("\nOverall average error: $%.3f\n", totalError);
        
        // Check prediction diversity
        Set<String> uniquePreds = new HashSet<>();
        for (float pred : allPredictions) {
            uniquePreds.add(String.format("%.2f", pred));
        }
        System.out.printf("Prediction diversity: %d unique values out of 1000\n", uniquePreds.size());
        
        // Memory usage
        long totalParams = totalInputParams + 
            firstLayerSize * (totalInputParams / features.length) +
            256 * firstLayerSize + 128 * 256 + 64 * 128 + 1 * 64;
        System.out.printf("Total model parameters: %,d (%.1f MB)\n",
            totalParams, totalParams * 4.0 / 1024 / 1024);
    }
}