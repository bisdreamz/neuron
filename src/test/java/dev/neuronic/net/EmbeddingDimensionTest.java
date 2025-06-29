package dev.neuronic.net;

import dev.neuronic.net.layers.Feature;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import dev.neuronic.net.simple.SimpleNet;
import dev.neuronic.net.simple.SimpleNetFloat;
import org.junit.jupiter.api.Test;

import java.util.*;

/**
 * Test different embedding dimensions to find optimal size for domains/bundles.
 * With only 30-50 active domains, we might not need many dimensions.
 */
public class EmbeddingDimensionTest {
    
    @Test
    public void testEmbeddingDimensions() {
        System.out.println("=== EMBEDDING DIMENSION COMPARISON ===\n");
        System.out.println("Testing with 40 hot domains (90% traffic) + cold tail\n");
        
        int[] dimensions = {2, 4, 8, 16, 32, 64};
        
        for (int dim : dimensions) {
            System.out.println("\n--- EMBEDDING DIM = " + dim + " ---");
            testWithDimension(dim);
        }
        
        System.out.println("\n\n=== SUMMARY ===");
        System.out.println("Embedding dimensions guidance:");
        System.out.println("- 2-4 dims: Too small, limited expressiveness");
        System.out.println("- 8 dims: Good for small sets (~50-100 active values)");
        System.out.println("- 16 dims: Sweet spot for medium sets (~100-1000 active values)");
        System.out.println("- 32 dims: Good for large sets or complex patterns");
        System.out.println("- 64+ dims: Probably overkill unless you have very complex interactions");
    }
    
    private void testWithDimension(int embDim) {
        AdamWOptimizer optimizer = new AdamWOptimizer(0.001f, 0.0f);
        
        Feature[] features = {
            Feature.embedding(5000, embDim, "domain"),
            Feature.oneHot(10, "format"),
            Feature.passthrough("bidfloor")
        };
        
        NeuralNet net = NeuralNet.newBuilder()
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(features))
            .layer(Layers.hiddenDenseLeakyRelu(64, 0.01f))
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        Random rand = new Random(42);
        
        // Create 40 hot domains
        Set<Integer> hotDomains = new HashSet<>();
        while (hotDomains.size() < 40) {
            hotDomains.add(rand.nextInt(200));
        }
        
        // Generate training data - similar to your throttle pattern
        List<Map<String, Object>> trainingData = new ArrayList<>();
        List<Float> targets = new ArrayList<>();
        
        for (int i = 0; i < 5000; i++) {
            Map<String, Object> input = new HashMap<>();
            
            if (rand.nextFloat() < 0.9f) {
                // 90% from hot domains
                int domain = new ArrayList<>(hotDomains).get(rand.nextInt(hotDomains.size()));
                input.put("domain", domain);
                
                // 1% bid rate for hot domains
                if (rand.nextFloat() < 0.01f) {
                    // Bid event - repeat 20x
                    float bidPrice = rand.nextFloat() * 4 + 1;
                    for (int j = 0; j < 20; j++) {
                        Map<String, Object> bidInput = new HashMap<>(input);
                        bidInput.put("format", rand.nextInt(10));
                        bidInput.put("bidfloor", rand.nextFloat() * 5);
                        trainingData.add(bidInput);
                        targets.add(bidPrice);
                    }
                } else {
                    // Auction only
                    input.put("format", rand.nextInt(10));
                    input.put("bidfloor", rand.nextFloat() * 5);
                    trainingData.add(input);
                    targets.add(-0.01f);
                }
            } else {
                // 10% from cold domains
                input.put("domain", rand.nextInt(4000) + 1000);
                input.put("format", rand.nextInt(10));
                input.put("bidfloor", rand.nextFloat() * 5);
                trainingData.add(input);
                targets.add(-0.01f);
            }
        }
        
        // Train
        long startTime = System.currentTimeMillis();
        for (int epoch = 0; epoch < 10; epoch++) {
            for (int i = 0; i < Math.min(2000, trainingData.size()); i++) {
                int idx = rand.nextInt(trainingData.size());
                model.train(trainingData.get(idx), targets.get(idx));
            }
        }
        long trainTime = System.currentTimeMillis() - startTime;
        
        // Test discrimination ability
        List<Float> hotPredictions = new ArrayList<>();
        List<Float> coldPredictions = new ArrayList<>();
        
        // Test hot domains
        for (int domain : new ArrayList<>(hotDomains).subList(0, 10)) {
            Map<String, Object> input = new HashMap<>();
            input.put("domain", domain);
            input.put("format", 0);
            input.put("bidfloor", 2.0f);
            hotPredictions.add(model.predictFloat(input));
        }
        
        // Test cold domains
        for (int i = 0; i < 10; i++) {
            Map<String, Object> input = new HashMap<>();
            input.put("domain", rand.nextInt(4000) + 1000);
            input.put("format", 0);
            input.put("bidfloor", 2.0f);
            coldPredictions.add(model.predictFloat(input));
        }
        
        // Calculate statistics
        float hotMean = (float) hotPredictions.stream().mapToDouble(Float::doubleValue).average().orElse(0);
        float coldMean = (float) coldPredictions.stream().mapToDouble(Float::doubleValue).average().orElse(0);
        float hotStd = calculateStd(hotPredictions, hotMean);
        float coldStd = calculateStd(coldPredictions, coldMean);
        
        // Test within-hot-domain discrimination
        Map<Integer, List<Float>> domainPreds = new HashMap<>();
        for (int domain : new ArrayList<>(hotDomains).subList(0, 5)) {
            List<Float> preds = new ArrayList<>();
            for (int format = 0; format < 5; format++) {
                Map<String, Object> input = new HashMap<>();
                input.put("domain", domain);
                input.put("format", format);
                input.put("bidfloor", 2.0f);
                preds.add(model.predictFloat(input));
            }
            domainPreds.put(domain, preds);
        }
        
        // Calculate within-domain variance
        float avgWithinDomainStd = 0;
        for (List<Float> preds : domainPreds.values()) {
            float mean = (float) preds.stream().mapToDouble(Float::doubleValue).average().orElse(0);
            avgWithinDomainStd += calculateStd(preds, mean);
        }
        avgWithinDomainStd /= domainPreds.size();
        
        // Calculate model size
        int numParams = embDim * 5000 + 10 + 1; // embeddings + one-hot + bidfloor
        
        System.out.printf("Hot domains: mean=%.3f, std=%.3f\n", hotMean, hotStd);
        System.out.printf("Cold domains: mean=%.3f, std=%.3f\n", coldMean, coldStd);
        System.out.printf("Hot-Cold separation: %.3f\n", Math.abs(hotMean - coldMean));
        System.out.printf("Within-domain discrimination (avg std): %.3f\n", avgWithinDomainStd);
        System.out.printf("Training time: %dms\n", trainTime);
        System.out.printf("Embedding parameters: %,d\n", embDim * 5000);
    }
    
    private float calculateStd(List<Float> values, float mean) {
        float variance = 0;
        for (float v : values) {
            variance += (v - mean) * (v - mean);
        }
        return (float) Math.sqrt(variance / values.size());
    }
}