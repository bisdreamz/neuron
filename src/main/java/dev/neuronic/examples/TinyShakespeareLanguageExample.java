package dev.neuronic.examples;

import dev.neuronic.net.Layers;
import dev.neuronic.net.NeuralNet;
import dev.neuronic.net.SamplingConfig;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import dev.neuronic.net.repl.LanguageModelREPL;
import dev.neuronic.net.simple.SimpleNet;
import dev.neuronic.net.simple.SimpleNetLanguageModel;
import dev.neuronic.net.simple.SimpleNetTrainingConfig;
import dev.neuronic.net.training.LearningRateSchedule;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Example demonstrating the TinyShakespeare language dataset. This demos automatic embeddings,
 * dictionary handling, training stats such as perplexity, saving the model, and finally
 * the REPL utility.
 */
public class TinyShakespeareLanguageExample {

    public static void main(String[] args) throws Exception {
        // hyperparams
        final int MAX_VOCAB_SZ   = 5_000;
        final int MAX_TOKENS     = 200_000;
        final int WINDOW_SIZE    = 20;
        final int EMBEDDING_SIZE = 32;
        final int HIDDEN_SIZE    = 128;
        final int BATCH_SIZE     = 128;
        final int EPOCHS         = 5;
        final float LEARNING_RATE = 0.000003f;

        // 1) load up to MAX_TOKENS tokens from our tiny file
        String[] tokens = new String[MAX_TOKENS];
        loadTokens(MAX_TOKENS, tokens);

        // 2) build our model
        SimpleNetLanguageModel lm = SimpleNet.ofLanguageModel(
                NeuralNet.newBuilder()
                        .input(WINDOW_SIZE)
                        .withGlobalGradientClipping(1f)
                        .setDefaultOptimizer(
                                new AdamWOptimizer(LEARNING_RATE, 0.00001f)
                        )
                        .layer(Layers.inputSequenceEmbedding(WINDOW_SIZE, MAX_VOCAB_SZ, EMBEDDING_SIZE))
                        .layer(Layers.hiddenGruLast(HIDDEN_SIZE))
                        .layer(Layers.hiddenDenseRelu(HIDDEN_SIZE))
                        .output(Layers.outputSoftmaxCrossEntropy(MAX_VOCAB_SZ))
        );

        // 3) assemble our bulk training data
        List<String[]>   inputs  = new ArrayList<>();
        List<String>     labels  = new ArrayList<>();
        for (int i = 0; i + WINDOW_SIZE < tokens.length; i++) {
            // stop if we hit a null (fewer tokens than MAX_TOKENS)
            if (tokens[i + WINDOW_SIZE] == null) break;
            inputs.add(Arrays.copyOfRange(tokens, i, i + WINDOW_SIZE));
            labels.add(tokens[i + WINDOW_SIZE]);
        }

        // 4) training config
        SimpleNetTrainingConfig config = SimpleNetTrainingConfig.builder()
                .batchSize(BATCH_SIZE)
                .epochs(EPOCHS)
                .shuffle(true)
                .verbosity(2)
                //.withLearningRateSchedule(LearningRateSchedule.exponentialDecay(LEARNING_RATE, 0.01f))
                .validationSplit(0.2f)
                //.withEarlyStopping(3)
                // .globalGradientClipNorm(1.0f)
                .build();

        // 5) train in bulk
        lm.trainBulk(inputs, labels, config).printSummary();

        // 6) drop into REPL
        lm.setSamplingConfig(SamplingConfig.temperature(0.95f));
        LanguageModelREPL.start(lm);
    }

    private static void loadTokens(int maxTokens, String[] tokens) throws Exception {
        AtomicInteger count = new AtomicInteger(0);
        var uri = TinyShakespeareLanguageExample.class.getResource("/language/ts/tiny.txt").toURI();
        for (String line : Files.readAllLines(Path.of(uri))) {
            if (line.isBlank()) continue;
            for (var tok : line.split("\\s+")) {
                int idx = count.getAndIncrement();
                if (idx >= maxTokens) return;
                tokens[idx] = tok.trim();
            }
        }
    }
}