package dev.neuronic.net;

import dev.neuronic.net.layers.Feature;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import dev.neuronic.net.simple.SimpleNet;
import dev.neuronic.net.simple.SimpleNetFloat;
import org.junit.jupiter.api.Test;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test using tiny positive values instead of negative penalties for auction costs.
 */
public class PositivePenaltyTest {
    
    @Test
    public void testPositiveVsNegativePenalties() {
        System.out.println("=== POSITIVE VS NEGATIVE PENALTY TEST ===\n");
        
        // Test both approaches
        testPenaltyApproach("Negative penalties (-$0.03)", -0.03f, false);
        testPenaltyApproach("Tiny positive values ($0.01)", 0.01f, true);
        testPenaltyApproach("Very tiny positive ($0.001)", 0.001f, true);
    }
    
    private void testPenaltyApproach(String name, float penaltyValue, boolean expectSuccess) {
        System.out.printf("\n=== Testing: %s ===\n", name);
        
        Feature[] features = {
            Feature.embeddingLRU(1000, 32, "ZONEID"),
            Feature.embeddingLRU(1000, 32, "DOMAIN"),
            Feature.passthrough("BIDFLOOR")
        };
        
        AdamWOptimizer optimizer = new AdamWOptimizer(0.001f, 0.001f);
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(features.length)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(features))
            .layer(Layers.hiddenDenseRelu(256))
            .layer(Layers.hiddenDenseRelu(128))
            .layer(Layers.hiddenDenseRelu(64))
            .withGlobalGradientClipping(1.0f)
            .output(Layers.outputHuberRegression(1, optimizer, 3.0f));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        Random rand = new Random(42);
        
        // Define premium segments (5%)
        Set<String> premiumSegments = new HashSet<>();
        for (int i = 0; i < 50; i++) {
            premiumSegments.add(i + "_" + i);
        }
        
        // Train with realistic distribution
        System.out.println("Training 50,000 requests...");
        int positiveBids = 0;
        int zeroBids = 0;
        
        for (int step = 0; step < 50_000; step++) {
            int zoneId = rand.nextInt(1000);
            int domainId = rand.nextInt(1000);
            String segment = zoneId + "_" + domainId;
            
            Map<String, Object> input = Map.of(
                "ZONEID", zoneId,
                "DOMAIN", domainId,
                "BIDFLOOR", 1.0f
            );
            
            // Step 1: Auction cost (using provided penalty value)
            model.train(input, penaltyValue);
            
            // Step 2: Bid result
            boolean isPremium = premiumSegments.contains(segment);
            float bidValue;
            
            if (isPremium && rand.nextFloat() < 0.8f) {
                bidValue = 2.0f + rand.nextFloat(); // $2-3 for premium
                positiveBids++;
            } else if (!isPremium && rand.nextFloat() < 0.05f) {
                bidValue = 0.2f + rand.nextFloat() * 0.3f; // $0.2-0.5 for regular
                positiveBids++;
            } else {
                bidValue = 0.0f; // No bid
                zeroBids++;
            }
            
            model.train(input, bidValue);
            
            // Check progress
            if (step > 0 && step % 10000 == 0) {
                float premiumPred = model.predictFloat(Map.of(
                    "ZONEID", 10, "DOMAIN", 10, "BIDFLOOR", 1.0f));
                float regularPred = model.predictFloat(Map.of(
                    "ZONEID", 500, "DOMAIN", 500, "BIDFLOOR", 1.0f));
                
                System.out.printf("Step %d: Premium=$%.3f, Regular=$%.3f\n",
                    step, premiumPred, regularPred);
            }
        }
        
        System.out.printf("Trained on %d positive bids, %d zero bids\n", positiveBids, zeroBids);
        
        // Final evaluation
        System.out.println("\nFinal predictions:");
        
        // Test multiple segments
        List<Float> premiumPreds = new ArrayList<>();
        List<Float> regularPreds = new ArrayList<>();
        
        for (int i = 0; i < 20; i++) {
            // Premium segments
            float pred = model.predictFloat(Map.of(
                "ZONEID", i, "DOMAIN", i, "BIDFLOOR", 1.0f));
            premiumPreds.add(pred);
            
            // Regular segments
            pred = model.predictFloat(Map.of(
                "ZONEID", 500 + i, "DOMAIN", 500 + i, "BIDFLOOR", 1.0f));
            regularPreds.add(pred);
        }
        
        float premiumAvg = premiumPreds.stream().reduce(0f, Float::sum) / premiumPreds.size();
        float regularAvg = regularPreds.stream().reduce(0f, Float::sum) / regularPreds.size();
        float difference = premiumAvg - regularAvg;
        
        System.out.printf("Premium average: $%.3f\n", premiumAvg);
        System.out.printf("Regular average: $%.3f\n", regularAvg);
        System.out.printf("Difference: $%.3f\n", difference);
        
        // Check uniqueness
        Set<String> uniquePremium = new HashSet<>();
        Set<String> uniqueRegular = new HashSet<>();
        for (float p : premiumPreds) uniquePremium.add(String.format("%.3f", p));
        for (float p : regularPreds) uniqueRegular.add(String.format("%.3f", p));
        
        System.out.printf("Unique predictions: %d premium, %d regular\n",
            uniquePremium.size(), uniqueRegular.size());
        
        boolean success = difference > 1.0f && premiumAvg > 1.5f && regularAvg < 0.5f;
        System.out.println(success ? "✓ SUCCESS - Model differentiates correctly!" :
                                    "❌ FAILURE - Model cannot differentiate!");
        
        if (expectSuccess) {
            assertTrue(success, "Model should differentiate with positive penalties");
        }
    }
}