package com.system.tensorhub.knn;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.Closeable;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.io.SequenceInputStream;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import com.system.tensorhub.knn.DataUtil.Row;
import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.google.common.base.Stopwatch;
import lombok.extern.slf4j.Slf4j;


/**
 * KnnSearch is a class for performing nearest neighbor search.
 */
@Slf4j
public class KnnSearch implements Closeable {

    /**
     * Constant representing the number of milliseconds in a second.
     */
    private static final int SEC_IN_MS = 1000;

    /**
     * Flag to indicate whether to show help information.
     */
    @Parameter(names = "--help", help = true)
    private boolean help;

    /**
     * List of data vector file names.
     */
    @Parameter(names = "--data-files", variableArity = true, description = "data vector file(s)", required = true)
    private List<String> dataFileNames = new ArrayList<>();

    /**
     * List of input vector file names.
     */
    @Parameter(names = "--input-files", variableArity = true, description = "input vector file(s)", required = true)
    private List<String> inputFileNames = new ArrayList<>();

    /**
     * File name to write the knn results to. stdout if not specified.
     */
    @Parameter(names = "--output-file", description = "file to write knn results to. stdout if not specified")
    private String outputFileName;

    /**
     * Number of nearest neighbors to find.
     */
    @Parameter(names = "--k", description = "k")
    private int k = 100;

    /**
     * Size of each batch.
     */
    @Parameter(names = "--batch-size", description = "batch size")
    private int batchSize = 128;

    /**
     * Type of data to load, either fp16 or fp32.
     */
    @Parameter(names = "--data-type", description = "loads the data as fp16 or fp32")
    private String dataType = "fp32";

    /**
     * Delimiter for separating key-vector on each row.
     */
    @Parameter(names = "--key-val-delim", description = "delimiter for separating key-vector on each row")
    private String keyValDelim = "\t";

    /**
     * Delimiter for separating vector elements on each row.
     */
    @Parameter(names = "--vec-delim", description = "delimiter for separating vector elements on each row")
    private String vectorDelim = " ";

    /**
     * Separator for id:score in the output.
     */
    @Parameter(names = "--id-score-sep", description = "separator for id:score in the output")
    private String idScoreSep = ":";

    /**
     * Number of decimal points for output score's precision.
     */
    @Parameter(names = "--score-precision", description = "number of decimal points for output score's precision")
    private int scorePrecision = 3;

    /**
     * Flag to indicate whether to output only keys and not scores.
     */
    @Parameter(names = "--output-keys-only", description = "do not output scores")
    private boolean outputKeysOnly;

    /**
     * Interval to print performance metrics after n batches.
     */
    @Parameter(names = "--report-interval", description = "print performance metrics after n batches")
    private int reportInterval = 1000;


    /**
     * The size of the feature.
     */
    private int featureSize;

    /**
     * The format for the score.
     */
    private String scoreFormat;

    /**
     * The executor service for the writer thread.
     */
    private ExecutorService writerThread;

    /**
     * The executor service for the search thread.
     */
    private ExecutorService searchThread;

    /**
     * The reader for the input data.
     */
    private BufferedReader reader;

    /**
     * The writer for the output data.
     */
    private PrintWriter writer;

    /**
     * The instance of KNearestNeighborsCuda.
     */
    private KNearestNeighborsCuda knnCuda;

    /**
     * Constructs a new KnnSearch object.
     *
     * @param args the command-line arguments
     * @throws IOException if an I/O error occurs
     */
    private KnnSearch(final String[] args) throws IOException {
        JCommander jc = new JCommander(this);
        jc.setProgramName("knn-search");

        try {
            jc.parse(args);

            if (help) {
                jc.usage();
                System.exit(0);
            }
        } catch (Exception e) {
            log.error("Error running command", e);
            jc.usage();
            System.exit(1);
        }

        String inputFile = inputFileNames.get(0);
        this.featureSize = DataUtil.findFeatureSize(new File(inputFile), keyValDelim, vectorDelim);
        log.info("Auto determined feature size = {} from file {}", featureSize, inputFile);

        this.scoreFormat = "%." + scorePrecision + "f";
        this.writerThread = Executors.newSingleThreadScheduledExecutor();
        this.searchThread = Executors.newSingleThreadScheduledExecutor();
        this.reader = createReader();
        this.writer = createWriter();

        this.knnCuda = new KNearestNeighborsCuda(k, batchSize, featureSize, DataType.fromString(this.dataType),
                toFile(dataFileNames), keyValDelim, vectorDelim);
        knnCuda.init();
    }

    /**
     * The main method that performs the search.
     *
     * @throws IOException if an I/O error occurs
     */
    private void run() throws IOException {
        log.info("Starting search. Reporting metrics every {} batches", reportInterval);

        Stopwatch timer = Stopwatch.createStarted();
        long totalBatchTime = 0;

        long batchNum = 0;
        String line = reader.readLine();
        while (line != null) {
            Stopwatch batchTimer = Stopwatch.createStarted();

            String[] inputRowIds = new String[batchSize];
            float[] inputVectors = new float[batchSize * featureSize];

            int i = 0;
            do {
                Row row = DataUtil.parseLine(line, keyValDelim, vectorDelim);
                inputRowIds[i] = row.key;
                float[] vector = row.vector;
                System.arraycopy(vector, 0, inputVectors, i * featureSize, featureSize);
                i++;
                line = reader.readLine();
            } while (i < batchSize && line != null);

            final int activeBatchSize = i;

            searchThread.submit(() -> {
                KnnResult result = knnCuda.findKnn(inputVectors);

                writerThread.submit(() -> {
                    for (int j = 0; j < activeBatchSize; j++) {
                        String inputRowId = inputRowIds[j];
                        writer.print(inputRowId);
                        writer.print(keyValDelim);

                        float score = result.getScoreAt(j, 0);
                        String key = result.getKeyAt(j, 0);
                        writer.print(key);
                        if (!outputKeysOnly) {
                            writer.print(idScoreSep);
                            writer.format(scoreFormat, score);
                        }

                        for (int m = 1; m < k; m++) {
                            score = result.getScoreAt(j, m);
                            key = result.getKeyAt(j, m);

                            writer.print(vectorDelim);
                            writer.print(key);
                            if (!outputKeysOnly) {
                                writer.print(idScoreSep);
                                writer.format(scoreFormat, score);
                            }
                        }
                        writer.println();
                    }
                });
            });

            long elapsedBatch = batchTimer.elapsed(TimeUnit.MILLISECONDS);
            long elapsedTotal = timer.elapsed(TimeUnit.SECONDS);
            totalBatchTime += elapsedBatch;

            ++batchNum;

            if (batchNum % reportInterval == 0) {
                log.info(String.format("Processed %7d batches in %4ds. Elapsed %7ds. TPS %7d", batchNum,
                        totalBatchTime / SEC_IN_MS, elapsedTotal,
                        (batchNum * batchSize) / timer.elapsed(TimeUnit.SECONDS)));
                totalBatchTime = 0;
            }
        }

        try {
            searchThread.shutdown();
            searchThread.awaitTermination(Long.MAX_VALUE, TimeUnit.MILLISECONDS);
            writerThread.shutdown();
            writerThread.awaitTermination(Long.MAX_VALUE, TimeUnit.MILLISECONDS);
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }

        long totalTime = timer.elapsed(TimeUnit.SECONDS);

        log.info("Done processing {} batches in {} s", batchNum, totalTime);
    }

    /**
     * Closes the reader, writer, and knnCuda.
     *
     * @throws IOException if an I/O error occurs
     */
    @Override
    public void close() throws IOException {
        reader.close();
        writer.close();
        knnCuda.close();
    }

    /**
     * Creates a reader for the input files.
     *
     * @return the created reader
     * @throws FileNotFoundException if the input file is not found
     */
    private BufferedReader createReader() throws FileNotFoundException {
        List<InputStream> fis = new ArrayList<>();
        for (String fileName : inputFileNames) {
            fis.add(new FileInputStream(fileName));
        }
        InputStream is = new SequenceInputStream(Collections.enumeration(fis));
        return new BufferedReader(new InputStreamReader(is, Charset.forName("UTF-8")));
    }

    /**
     * Creates a writer for the output file.
     *
     * @return the created writer
     * @throws IOException if an I/O error occurs
     */
    private PrintWriter createWriter() throws IOException {
        OutputStream os;
        if (outputFileName == null) {
            os = System.out;
        } else {
            File outputFile = new File(outputFileName);
            outputFile.getParentFile().mkdirs();
            outputFile.createNewFile();
            os = new FileOutputStream(outputFileName);
        }
        return new PrintWriter(new BufferedWriter(new OutputStreamWriter(os, Charset.forName("UTF-8"))));
    }

    /**
     * Converts a list of file names to a list of files.
     *
     * @param fileNames the list of file names to convert
     * @return the list of converted files
     */
    private List<File> toFile(final List<String> fileNames) {
        List<File> files = new ArrayList<>();
        for (String fileName : fileNames) {
            files.add(new File(fileName));
        }
        return files;
    }

    /**
     * The main method that performs the search.
     *
     * @param args the command-line arguments
     */
    public static void main(final String[] args) {
        try (KnnSearch knnSearch = new KnnSearch(args)) {
            knnSearch.run();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}
