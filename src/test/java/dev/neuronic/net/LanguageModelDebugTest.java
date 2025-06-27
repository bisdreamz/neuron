package dev.neuronic.net;

import dev.neuronic.net.layers.InputSequenceEmbeddingLayer;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import dev.neuronic.net.simple.SimpleNet;
import dev.neuronic.net.simple.SimpleNetLanguageModel;
import dev.neuronic.net.simple.SimpleNetTrainingConfig;
import org.junit.jupiter.api.Test;

import java.util.*;

/**
 * Debug test to understand why language models aren't learning.
 */
public class LanguageModelDebugTest {
    
    @Test
    public void testVocabularyBuilding() {
        // Build a minimal model
        SimpleNetLanguageModel model = SimpleNet.ofLanguageModel(
            NeuralNet.newBuilder()
                .input(3)
                .setDefaultOptimizer(new AdamWOptimizer(0.1f, 0.0f))
                .layer(Layers.inputSequenceEmbedding(3, 10, 8))
                .layer(Layers.hiddenDenseRelu(8))
                .output(Layers.outputSoftmaxCrossEntropy(10))
        );
        
        InputSequenceEmbeddingLayer embedLayer = 
            (InputSequenceEmbeddingLayer) model.getNetwork().getInputLayer();
        
        // Check initial vocabulary
        System.out.println("Initial vocab size: " + embedLayer.getVocabularySize());
        
        // Create simple data
        List<String[]> sequences = Arrays.asList(
            new String[]{"a", "a", "a"},
            new String[]{"b", "b", "b"}
        );
        List<String> targets = Arrays.asList("b", "a");
        
        // Check token IDs before training
        System.out.println("\nBefore training:");
        System.out.println("Token ID for 'a': " + embedLayer.getTokenId("a"));
        System.out.println("Token ID for 'b': " + embedLayer.getTokenId("b"));
        System.out.println("Token ID for '<unk>': " + embedLayer.getTokenId("<unk>"));
        System.out.println("Vocab size: " + embedLayer.getVocabularySize());
        
        // Train for one epoch
        SimpleNetTrainingConfig config = SimpleNetTrainingConfig.builder()
            .batchSize(2)
            .epochs(1)
            .validationSplit(0.0f)
            .verbosity(0)
            .build();
            
        model.trainBulk(sequences, targets, config);
        
        // Check token IDs after training
        System.out.println("\nAfter training:");
        System.out.println("Token ID for 'a': " + embedLayer.getTokenId("a"));
        System.out.println("Token ID for 'b': " + embedLayer.getTokenId("b"));
        System.out.println("Vocab size: " + embedLayer.getVocabularySize());
        
        // Check tokenization
        System.out.println("\nTokenization check:");
        float[] tokensA = new float[3];
        float[] tokensB = new float[3];
        for (int i = 0; i < 3; i++) {
            tokensA[i] = embedLayer.getTokenId("a");
            tokensB[i] = embedLayer.getTokenId("b");
        }
        System.out.println("Tokens for 'a a a': " + Arrays.toString(tokensA));
        System.out.println("Tokens for 'b b b': " + Arrays.toString(tokensB));
        
        // Test predictions
        float[] probsA = model.predictProbabilities(new String[]{"a", "a", "a"});
        float[] probsB = model.predictProbabilities(new String[]{"b", "b", "b"});
        
        System.out.println("\nProbabilities for 'a a a':");
        float sumA = 0;
        for (int i = 0; i < probsA.length; i++) {
            System.out.printf("  Class %d (%s): %.4f\n", i, embedLayer.getWord(i), probsA[i]);
            sumA += probsA[i];
        }
        System.out.println("  Sum: " + sumA);
        
        System.out.println("\nProbabilities for 'b b b':");
        float sumB = 0;
        for (int i = 0; i < probsB.length; i++) {
            System.out.printf("  Class %d (%s): %.4f\n", i, embedLayer.getWord(i), probsB[i]);
            sumB += probsB[i];
        }
        System.out.println("  Sum: " + sumB);
        
        // Check the actual predictions
        String predA = model.predictNext(new String[]{"a", "a", "a"});
        String predB = model.predictNext(new String[]{"b", "b", "b"});
        System.out.println("\nPredictions:");
        System.out.println("'a a a' -> '" + predA + "'");
        System.out.println("'b b b' -> '" + predB + "'");
        
        // Check if embeddings are different
        float[] embA = embedLayer.getWordEmbedding("a");
        float[] embB = embedLayer.getWordEmbedding("b");
        System.out.println("\nEmbeddings:");
        System.out.println("Embedding for 'a': " + Arrays.toString(Arrays.copyOf(embA, 4)) + "...");
        System.out.println("Embedding for 'b': " + Arrays.toString(Arrays.copyOf(embB, 4)) + "...");
        
        // Check if they're identical
        boolean identical = true;
        for (int i = 0; i < embA.length; i++) {
            if (Math.abs(embA[i] - embB[i]) > 1e-6) {
                identical = false;
                break;
            }
        }
        System.out.println("Embeddings identical? " + identical);
        
        // Test the network directly with token IDs
        System.out.println("\nDirect network test:");
        float[] directProbsA = model.getNetwork().predict(tokensA);
        float[] directProbsB = model.getNetwork().predict(tokensB);
        System.out.println("Direct probs for [1,1,1]: " + Arrays.toString(Arrays.copyOf(directProbsA, 5)) + "...");
        System.out.println("Direct probs for [2,2,2]: " + Arrays.toString(Arrays.copyOf(directProbsB, 5)) + "...");
    }
    
    @Test
    public void testOneHotEncoding() {
        // Check if one-hot encoding is correct
        SimpleNetLanguageModel model = SimpleNet.ofLanguageModel(
            NeuralNet.newBuilder()
                .input(3)
                .setDefaultOptimizer(new AdamWOptimizer(0.1f, 0.0f))
                .layer(Layers.inputSequenceEmbedding(3, 10, 8))
                .output(Layers.outputSoftmaxCrossEntropy(10))
        );
        
        // Ensure vocabulary is built
        model.predictNext(new String[]{"a", "b", "c"});
        
        InputSequenceEmbeddingLayer embedLayer = 
            (InputSequenceEmbeddingLayer) model.getNetwork().getInputLayer();
        
        System.out.println("\nChecking one-hot encoding:");
        String[] testWords = {"<unk>", "a", "b", "c"};
        for (String word : testWords) {
            int tokenId = embedLayer.getTokenId(word);
            System.out.printf("Word '%s' -> Token ID %d\n", word, tokenId);
        }
    }
}