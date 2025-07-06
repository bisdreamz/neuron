package dev.neuronic.net;

import dev.neuronic.net.layers.InputSequenceEmbeddingLayer;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import dev.neuronic.net.simple.SimpleNet;
import dev.neuronic.net.simple.SimpleNetLanguageModel;
import dev.neuronic.net.simple.SimpleNetTrainingConfig;
import dev.neuronic.net.simple.SimpleNetTrainingResult;
import org.junit.jupiter.api.Test;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * End-to-end test for language modeling on tiny, predictable datasets.
 * Verifies the model can learn simple patterns and achieve very low perplexity.
 */
public class SimpleLanguageModelTest {
    
    @Test
    public void testCanLearnSimpleRepetitivePattern() {
        // Create a tiny dataset with a simple repeating pattern: "a b c a b c ..."
        List<String> pattern = Arrays.asList("a", "b", "c");
        List<String[]> sequences = new ArrayList<>();
        List<String> targets = new ArrayList<>();
        
        // Generate 20 sequences of length 3 (reduced from 100 for faster convergence)
        for (int i = 0; i < 20; i++) {
            for (int j = 0; j < pattern.size(); j++) {
                String[] seq = new String[3];
                for (int k = 0; k < 3; k++) {
                    seq[k] = pattern.get((i * 3 + j + k) % pattern.size());
                }
                sequences.add(seq);
                targets.add(pattern.get((i * 3 + j + 3) % pattern.size()));
            }
        }
        
        // Build small model with dense layers instead of GRU
        SimpleNetLanguageModel model = SimpleNet.ofLanguageModel(
            NeuralNet.newBuilder()
                .input(3)
                .setDefaultOptimizer(new AdamWOptimizer(0.1f, 0.0f)) // Higher LR, no weight decay
                .withSeed(12345L)  // Use fixed seed for deterministic test
                .layer(Layers.inputSequenceEmbedding(3, 10, 16)) // Larger embedding
                .layer(Layers.hiddenDenseRelu(32)) // Dense layers work better for simple patterns
                .layer(Layers.hiddenDenseRelu(16))
                .output(Layers.outputSoftmaxCrossEntropy(10))
        );
        
        // Train
        SimpleNetTrainingConfig config = SimpleNetTrainingConfig.builder()
            .batchSize(10) // Even smaller batch size
            .epochs(100) // More epochs for better convergence
            .shuffle(false) // Keep pattern order
            .verbosity(0)
            .build();
            
        model.trainBulk(sequences, targets, config);
        
        // Test predictions
        
        // After seeing "a b c", should predict "a" with very high confidence
        String[] testSeq1 = {"a", "b", "c"};
        String pred1 = model.predictNext(testSeq1);
        assertEquals("a", pred1, "Model should predict 'a' after 'a b c'");
        
        // After seeing "b c a", should predict "b"
        String[] testSeq2 = {"b", "c", "a"};
        String pred2 = model.predictNext(testSeq2);
        assertEquals("b", pred2, "Model should predict 'b' after 'b c a'");
    }
    
    @Test 
    public void testCanLearnDeterministicSequence() {
        // Even simpler: "the cat sat on the mat" repeated
        String[] sentence = {"the", "cat", "sat", "on", "the", "mat"};
        List<String[]> sequences = new ArrayList<>();
        List<String> targets = new ArrayList<>();
        
        // Generate sequences by sliding window
        for (int epoch = 0; epoch < 50; epoch++) {
            for (int i = 0; i < sentence.length; i++) {
                String[] seq = new String[3];
                for (int j = 0; j < 3; j++) {
                    seq[j] = sentence[(i + j) % sentence.length];
                }
                sequences.add(seq);
                targets.add(sentence[(i + 3) % sentence.length]);
            }
        }
        
        // Build model
        SimpleNetLanguageModel model = SimpleNet.ofLanguageModel(
            NeuralNet.newBuilder()
                .input(3)
                .setDefaultOptimizer(new AdamWOptimizer(0.1f, 0.0001f)) // High LR for fast convergence
                .withSeed(12345L)  // Use fixed seed for deterministic test
                .layer(Layers.inputSequenceEmbedding(3, 10, 16))
                .layer(Layers.hiddenDenseRelu(16))
                .output(Layers.outputSoftmaxCrossEntropy(10))
        );
        
        // Train
        SimpleNetTrainingConfig config = SimpleNetTrainingConfig.builder()
            .batchSize(6)
            .epochs(20)
            .shuffle(false)
            .verbosity(0)
            .build();
            
        SimpleNetTrainingResult result = model.trainBulk(sequences, targets, config);
        
        // Model should achieve near-zero loss
        float finalLoss = result.getFinalLoss();
        assertTrue(finalLoss < 0.1f, "Final loss should be near zero for deterministic sequence, got: " + finalLoss);
        
        // Test generation
        
        // Starting from "the cat sat", should generate "on the mat"
        List<String> generated = new ArrayList<>(Arrays.asList("the", "cat", "sat"));
        for (int i = 0; i < 3; i++) {
            String[] context = generated.subList(generated.size() - 3, generated.size()).toArray(new String[0]);
            String next = model.predictNext(context);
            generated.add(next);
        }
        
        String output = String.join(" ", generated.subList(3, 6));
        assertEquals("on the mat", output, "Model should complete the memorized sequence");
        
        // Perplexity should be near 1 (perfect prediction)
        float perplexity = calculatePerplexity(model, sequences, targets);
        assertTrue(perplexity < 1.1f, "Perplexity should be near 1 for memorized sequence, got: " + perplexity);
    }
    
    @Test
    public void testFailsOnRandomData() {
        // Verify model can't learn truly random data (sanity check)
        Random rand = new Random(42);
        String[] vocab = {"a", "b", "c", "d", "e"};
        List<String[]> sequences = new ArrayList<>();
        List<String> targets = new ArrayList<>();
        
        for (int i = 0; i < 100; i++) {
            String[] seq = new String[3];
            for (int j = 0; j < 3; j++) {
                seq[j] = vocab[rand.nextInt(vocab.length)];
            }
            sequences.add(seq);
            targets.add(vocab[rand.nextInt(vocab.length)]);
        }
        
        SimpleNetLanguageModel model = SimpleNet.ofLanguageModel(
            NeuralNet.newBuilder()
                .input(3)
                .setDefaultOptimizer(new AdamWOptimizer(0.01f, 0.0001f))
                .withSeed(12345L)  // Use fixed seed for deterministic test
                .layer(Layers.inputSequenceEmbedding(3, 10, 8))
                .layer(Layers.hiddenGruLastNormalized(8))
                .output(Layers.outputSoftmaxCrossEntropy(10))
        );
        
        SimpleNetTrainingConfig config = SimpleNetTrainingConfig.builder()
            .batchSize(32)
            .epochs(5)
            .verbosity(0)
            .build();
            
        model.trainBulk(sequences, targets, config);
        
        // Perplexity should remain high for random data
        float perplexity = calculatePerplexity(model, sequences, targets);
        assertTrue(perplexity > 3.0f, "Perplexity should remain high for random data, got: " + perplexity);
    }
    
    @Test
    public void testDenseOnlyCanLearnSimplePattern() {
        // Test with only dense layers - no GRU
        List<String[]> sequences = new ArrayList<>();
        List<String> targets = new ArrayList<>();
        
        // Very simple pattern: after seeing "a a a", predict "b"
        // after seeing "b b b", predict "a"
        for (int i = 0; i < 100; i++) {
            sequences.add(new String[]{"a", "a", "a"});
            targets.add("b");
            sequences.add(new String[]{"b", "b", "b"});
            targets.add("a");
        }
        
        // Build model with only dense layers
        SimpleNetLanguageModel model = SimpleNet.ofLanguageModel(
            NeuralNet.newBuilder()
                .input(3)
                .setDefaultOptimizer(new AdamWOptimizer(0.1f, 0.0f)) // No weight decay
                .withSeed(12345L)  // Use fixed seed for deterministic test
                .layer(Layers.inputSequenceEmbedding(3, 10, 16))
                .layer(Layers.hiddenDenseRelu(32))
                .layer(Layers.hiddenDenseRelu(16))
                .output(Layers.outputSoftmaxCrossEntropy(10))
        );
        
        // Train
        SimpleNetTrainingConfig config = SimpleNetTrainingConfig.builder()
            .batchSize(20)
            .epochs(50)
            .shuffle(false)
            .verbosity(0)
            .build();
            
        SimpleNetTrainingResult result = model.trainBulk(sequences, targets, config);
        float finalLoss = result.getFinalLoss();
        
        // Should learn this simple pattern
        assertTrue(finalLoss < 0.5f, "Dense-only model should learn simple pattern, loss: " + finalLoss);
        
        // Test predictions
        String pred1 = model.predictNext(new String[]{"a", "a", "a"});
        assertEquals("b", pred1, "Should predict 'b' after 'a a a'");
        
        String pred2 = model.predictNext(new String[]{"b", "b", "b"});
        assertEquals("a", pred2, "Should predict 'a' after 'b b b'");
    }
    
    @Test
    public void testGradientFlowInSimpleModel() {
        // Create minimal model to verify gradients flow
        NeuralNet net = NeuralNet.newBuilder()
            .input(3)
            .setDefaultOptimizer(new AdamWOptimizer(0.1f, 0.0f))
            .withSeed(42L)  // Use fixed seed for deterministic test
            .layer(Layers.inputSequenceEmbedding(3, 5, 8))
            .layer(Layers.hiddenDenseRelu(8))
            .output(Layers.outputSoftmaxCrossEntropy(5));
            
        SimpleNetLanguageModel model = SimpleNet.ofLanguageModel(net);
        
        // Single training example
        String[][] sequences = {{"a", "b", "c"}};
        String[] targets = {"d"};
        
        // Get initial loss
        float[] probs1 = model.predictProbabilities(sequences[0]);
        InputSequenceEmbeddingLayer embedLayer = (InputSequenceEmbeddingLayer) model.getNetwork().getInputLayer();
        int targetId = embedLayer.getTokenId(targets[0]);
        float initialLoss = -1f * (float)Math.log(Math.max(probs1[targetId], 1e-7f));
        
        // Train for one epoch
        SimpleNetTrainingConfig config = SimpleNetTrainingConfig.builder()
            .batchSize(1)
            .epochs(1)
            .verbosity(0)
            .build();
            
        model.trainBulk(Arrays.asList(sequences), Arrays.asList(targets), config);
        
        // Get loss after training
        float[] probs2 = model.predictProbabilities(sequences[0]);
        float finalLoss = -1f * (float)Math.log(Math.max(probs2[targetId], 1e-7f));
        
        // Loss should decrease
        assertTrue(finalLoss < initialLoss, 
            String.format("Loss should decrease after training. Initial: %.4f, Final: %.4f", initialLoss, finalLoss));
    }
    
    private float calculatePerplexity(SimpleNetLanguageModel model, List<String[]> sequences, List<String> targets) {
        InputSequenceEmbeddingLayer embedLayer = (InputSequenceEmbeddingLayer) model.getNetwork().getInputLayer();
        
        float totalLogProb = 0;
        int count = 0;
        
        for (int i = 0; i < Math.min(sequences.size(), 50); i++) {
            float[] probs = model.predictProbabilities(sequences.get(i));
            int targetId = embedLayer.getTokenId(targets.get(i));
            
            if (targetId >= 0 && targetId < probs.length) {
                float prob = Math.max(probs[targetId], 1e-7f);
                totalLogProb += Math.log(prob);
                count++;
            }
        }
        
        return (float) Math.exp(-totalLogProb / count);
    }
}