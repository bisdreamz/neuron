package dev.neuronic.net;

import dev.neuronic.net.layers.Feature;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import dev.neuronic.net.simple.SimpleNet;
import dev.neuronic.net.simple.SimpleNetFloat;
import org.junit.jupiter.api.Test;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test a balanced training approach where auction penalties are offset
 * by the actual bid values to prevent collapse.
 */
public class BalancedTrainingTest {
    
    @Test
    public void testBalancedTrainingApproach() {
        System.out.println("=== BALANCED TRAINING TEST ===\n");
        System.out.println("Testing different approaches to handle auction penalties:\n");
        
        // Test different approaches
        testApproach("Separate penalty training", true, false);
        testApproach("Combined penalty+bid training", false, true);
        testApproach("No penalty (baseline)", false, false);
    }
    
    private void testApproach(String name, boolean separatePenalty, boolean combinedTraining) {
        System.out.printf("\n=== %s ===\n", name);
        
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
        
        System.out.println("Training 50,000 requests...");
        
        for (int step = 0; step < 50_000; step++) {
            // Weighted selection: 30% premium, 70% regular
            String segment;
            int zoneId, domainId;
            
            if (rand.nextFloat() < 0.3f) {
                // Premium segment
                zoneId = rand.nextInt(50);
                domainId = zoneId;
                segment = zoneId + "_" + domainId;
            } else {
                // Regular segment
                zoneId = 50 + rand.nextInt(950);
                domainId = 50 + rand.nextInt(950);
                segment = zoneId + "_" + domainId;
            }
            
            Map<String, Object> input = Map.of(
                "ZONEID", zoneId,
                "DOMAIN", domainId,
                "BIDFLOOR", 1.0f
            );
            
            boolean isPremium = premiumSegments.contains(segment);
            
            // Determine bid outcome
            float bidValue;
            if (isPremium && rand.nextFloat() < 0.8f) {
                bidValue = 2.0f + rand.nextFloat(); // $2-3 for premium
            } else if (!isPremium && rand.nextFloat() < 0.05f) {
                bidValue = 0.2f + rand.nextFloat() * 0.3f; // $0.2-0.5 for regular
            } else {
                bidValue = 0.0f; // No bid
            }
            
            // Apply training strategy
            if (separatePenalty) {
                // Train penalty and bid separately
                model.train(input, -0.03f); // Auction penalty
                model.train(input, bidValue); // Actual bid
            } else if (combinedTraining) {
                // Train with combined value (bid minus penalty)
                float combinedValue = bidValue - 0.03f;
                model.train(input, combinedValue);
            } else {
                // No penalty, just bid value
                model.train(input, bidValue);
            }
            
            // Progress check
            if (step > 0 && step % 10000 == 0) {
                float premiumPred = model.predictFloat(Map.of(
                    "ZONEID", 10, "DOMAIN", 10, "BIDFLOOR", 1.0f));
                float regularPred = model.predictFloat(Map.of(
                    "ZONEID", 500, "DOMAIN", 500, "BIDFLOOR", 1.0f));
                
                System.out.printf("Step %d: Premium=$%.3f, Regular=$%.3f, Diff=$%.3f\n",
                    step, premiumPred, regularPred, premiumPred - regularPred);
            }
        }
        
        // Final evaluation
        System.out.println("\nFinal predictions:");
        evaluatePredictions(model, premiumSegments);
    }
    
    private void evaluatePredictions(SimpleNetFloat model, Set<String> premiumSegments) {
        List<Float> premiumPreds = new ArrayList<>();
        List<Float> regularPreds = new ArrayList<>();
        
        // Test premium segments
        for (int i = 0; i < 20; i++) {
            float pred = model.predictFloat(Map.of(
                "ZONEID", i, "DOMAIN", i, "BIDFLOOR", 1.0f));
            premiumPreds.add(pred);
        }
        
        // Test regular segments
        for (int i = 500; i < 520; i++) {
            float pred = model.predictFloat(Map.of(
                "ZONEID", i, "DOMAIN", i, "BIDFLOOR", 1.0f));
            regularPreds.add(pred);
        }
        
        float premiumAvg = premiumPreds.stream().reduce(0f, Float::sum) / premiumPreds.size();
        float regularAvg = regularPreds.stream().reduce(0f, Float::sum) / regularPreds.size();
        float difference = premiumAvg - regularAvg;
        
        // Check uniqueness
        Set<String> uniquePremium = new HashSet<>();
        Set<String> uniqueRegular = new HashSet<>();
        for (float p : premiumPreds) uniquePremium.add(String.format("%.3f", p));
        for (float p : regularPreds) uniqueRegular.add(String.format("%.3f", p));
        
        System.out.printf("Premium: avg=$%.3f (expected ~$2.50), %d unique values\n",
            premiumAvg, uniquePremium.size());
        System.out.printf("Regular: avg=$%.3f (expected ~$0.25), %d unique values\n",
            regularAvg, uniqueRegular.size());
        System.out.printf("Differentiation: $%.3f\n", difference);
        
        boolean success = difference > 1.0f && premiumAvg > 1.5f && uniquePremium.size() > 10;
        System.out.println(success ? "\n✓ SUCCESS - Model differentiates segments!" :
                                    "\n❌ FAILURE - Model cannot differentiate!");
    }
}