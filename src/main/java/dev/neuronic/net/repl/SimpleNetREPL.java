package dev.neuronic.net.repl;

import java.util.Scanner;

/**
 * Generic REPL (Read-Eval-Print Loop) for interactive model testing.
 * Works with any model type through configurable transformers and formatters.
 * 
 * <p>Example usage for classification:
 * <pre>{@code
 * SimpleNetREPL<float[], Integer> repl = new SimpleNetREPL<>(
 *     model,
 *     input -> parseFloatArray(input),
 *     output -> "Predicted class: " + output
 * );
 * repl.run();
 * }</pre>
 * 
 * <p>Example usage for language models:
 * <pre>{@code
 * SimpleNetREPL<String[], String> repl = new SimpleNetREPL<>(
 *     model,
 *     input -> input.split(" "),
 *     output -> output,
 *     (m, input, max) -> generateUntilPunctuation(m, input)
 * );
 * repl.run();
 * }</pre>
 * 
 * @param <I> input type
 * @param <O> output type
 */
public class SimpleNetREPL<I, O> {
    
    private final Object model;
    private final InputTransformer<I> inputTransformer;
    private final OutputFormatter<O> outputFormatter;
    private final ResponseGenerator<I, O> responseGenerator;
    private final ModelPredictor<I, O> predictor;
    
    private String prompt = "> ";
    private String welcomeMessage = "=== Interactive Model Testing ===\nType 'quit' to exit, 'help' for options.\n";
    private boolean showTiming = false;
    
    /**
     * Functional interface for model prediction.
     */
    @FunctionalInterface
    public interface ModelPredictor<I, O> {
        O predict(Object model, I input);
    }
    
    /**
     * Create a REPL with basic prediction (no response generation).
     */
    public SimpleNetREPL(Object model, 
                        InputTransformer<I> inputTransformer,
                        OutputFormatter<O> outputFormatter,
                        ModelPredictor<I, O> predictor) {
        this(model, inputTransformer, outputFormatter, null, predictor);
    }
    
    /**
     * Create a REPL with response generation support.
     */
    public SimpleNetREPL(Object model,
                        InputTransformer<I> inputTransformer,
                        OutputFormatter<O> outputFormatter,
                        ResponseGenerator<I, O> responseGenerator,
                        ModelPredictor<I, O> predictor) {
        this.model = model;
        this.inputTransformer = inputTransformer;
        this.outputFormatter = outputFormatter;
        this.responseGenerator = responseGenerator;
        this.predictor = predictor;
    }
    
    /**
     * Set custom prompt.
     */
    public SimpleNetREPL<I, O> withPrompt(String prompt) {
        this.prompt = prompt;
        return this;
    }
    
    /**
     * Set custom welcome message.
     */
    public SimpleNetREPL<I, O> withWelcomeMessage(String message) {
        this.welcomeMessage = message;
        return this;
    }
    
    /**
     * Enable timing information.
     */
    public SimpleNetREPL<I, O> withTiming(boolean showTiming) {
        this.showTiming = showTiming;
        return this;
    }
    
    /**
     * Run the interactive REPL.
     */
    public void run() {
        Scanner scanner = new Scanner(System.in);
        System.out.println(welcomeMessage);
        
        while (true) {
            System.out.print(prompt);
            String input = scanner.nextLine().trim();
            
            if (input.equalsIgnoreCase("quit") || input.equalsIgnoreCase("exit")) {
                break;
            }
            
            if (input.equalsIgnoreCase("help")) {
                printHelp();
                continue;
            }
            
            if (input.equalsIgnoreCase("timing on")) {
                showTiming = true;
                System.out.println("Timing enabled.");
                continue;
            }
            
            if (input.equalsIgnoreCase("timing off")) {
                showTiming = false;
                System.out.println("Timing disabled.");
                continue;
            }
            
            if (input.isEmpty()) {
                continue;
            }
            
            try {
                processInput(input);
            } catch (Exception e) {
                System.err.println("Error: " + e.getMessage());
            }
        }
        
        scanner.close();
        System.out.println("\nGoodbye!");
    }
    
    private void processInput(String input) {
        long startTime = System.currentTimeMillis();
        
        // Transform input
        I transformed = inputTransformer.transform(input);
        
        // Get prediction or generate response
        O output;
        if (responseGenerator != null && input.toLowerCase().startsWith("generate:")) {
            // Use response generator if available and requested
            String actualInput = input.substring(9).trim();
            I genInput = inputTransformer.transform(actualInput);
            output = responseGenerator.generateResponse(model, genInput, 100);
        } else {
            // Standard prediction
            output = predictor.predict(model, transformed);
        }
        
        // Format and display output
        String formatted = outputFormatter.format(output);
        System.out.println(formatted);
        
        if (showTiming) {
            long elapsed = System.currentTimeMillis() - startTime;
            System.out.printf("Time: %d ms\n", elapsed);
        }
        
        System.out.println();
    }
    
    private void printHelp() {
        System.out.println("\nCommands:");
        System.out.println("  quit/exit     - Exit the REPL");
        System.out.println("  help          - Show this help");
        System.out.println("  timing on/off - Toggle timing information");
        if (responseGenerator != null) {
            System.out.println("  generate:<input> - Generate extended response");
        }
        System.out.println("\nEnter your input in the expected format for the model.");
        System.out.println();
    }
}