import org.opencv.core.Core;
import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.*;
import java.math.BigDecimal;
import java.math.MathContext;
import java.util.*;


public class PR {

    static { System.loadLibrary(Core.NATIVE_LIBRARY_NAME); }

    private static final ArrayList<BufferedImage> bufferedImages = new ArrayList<>();

    private static final ArrayList<ArrayList<BigDecimal[][]>> trainingPixelsImages = new ArrayList<>();
    private static byte[] trainingTrueFalse;

    private static final ArrayList<ArrayList<BigDecimal[][]>> testingPixelsImages = new ArrayList<>();
    private static byte[] testingTrueFalse;

    private static BufferedImage dimg_resize;
    private static BufferedImage image_listFilesForFolder;
    private static ArrayList<Integer> result_testing = new ArrayList<>();

    private static final MathContext mathContext20 = new MathContext(20);

    public static void main(String[] args) {

        /* 0 - 255
             *
            System.out.println(new Color(buf.getRGB(i, j)).getRed());
            System.out.println(new Color(buf.getRGB(i, j)).getGreen());
            System.out.println(new Color(buf.getRGB(i, j)).getBlue());*/

        int bufferedImagesSize;
        BigDecimal[][] pixels;
        File folder;

        ArrayList<BigDecimal[][]> tmp_trainingPixelsImage;
        folder = new File("not apples/" + 0);
        listFilesForFolder(folder, bufferedImages);
        bufferedImagesSize = bufferedImages.size();
        System.out.println("0 bufferedImages0.size: " + bufferedImagesSize);
        trainingTrueFalse = new byte[bufferedImagesSize*2]; // кол-во изображений TRUE должно равняться кол-ву изображений FALSE
        for (int i = bufferedImagesSize - 1; i >= 0; i--) {

            tmp_trainingPixelsImage = new ArrayList<>();

            pixels = new BigDecimal[bufferedImages.get(i).getHeight()][];
            for (int h = 0; h < bufferedImages.get(i).getHeight(); h++) {
                pixels[h] = new BigDecimal[bufferedImages.get(i).getWidth()];
                for (int w = 0; w < bufferedImages.get(i).getWidth(); w++)
                    pixels[h][w] = BigDecimal.valueOf( ((double) new Color(bufferedImages.get(i).getRGB(h, w)).getRed())/255. );
            }
//            pixels.add(1); // смещение

            tmp_trainingPixelsImage.add(pixels);

            pixels = new BigDecimal[bufferedImages.get(i).getHeight()][];
            for (int h = 0; h < bufferedImages.get(i).getHeight(); h++) {
                pixels[h] = new BigDecimal[bufferedImages.get(i).getWidth()];
                for (int w = 0; w < bufferedImages.get(i).getWidth(); w++)
                    pixels[h][w] = BigDecimal.valueOf( ((double) new Color(bufferedImages.get(i).getRGB(h, w)).getGreen())/255. );
            }
//            pixels.add(1); // смещение

            tmp_trainingPixelsImage.add(pixels);

            pixels = new BigDecimal[bufferedImages.get(i).getHeight()][];
            for (int h = 0; h < bufferedImages.get(i).getHeight(); h++) {
                pixels[h] = new BigDecimal[bufferedImages.get(i).getWidth()];
                for (int w = 0; w < bufferedImages.get(i).getWidth(); w++)
                    pixels[h][w] = BigDecimal.valueOf( ((double) new Color(bufferedImages.get(i).getRGB(h, w)).getBlue())/255. );
            }
//            pixels.add(1); // смещение

            tmp_trainingPixelsImage.add(pixels);

            trainingPixelsImages.add(tmp_trainingPixelsImage);
            trainingTrueFalse[-i + bufferedImagesSize - 1] = (byte) 0;

            bufferedImages.get(i).flush();
            bufferedImages.remove(bufferedImages.size() - 1);
        }
        bufferedImages.clear();
        System.out.println("0 bufferedImages0.size: 0");

        folder = new File("apples/" + 0);
        listFilesForFolder(folder, bufferedImages);
        System.out.println("1 bufferedImages0.size: " + bufferedImages.size());
        bufferedImagesSize = bufferedImages.size();
//        trainingTrueFalse = new byte[bufferedImagesSize];
        for (int i = bufferedImagesSize - 1; i >= bufferedImagesSize/2; i--) {

            tmp_trainingPixelsImage = new ArrayList<>();

            pixels = new BigDecimal[bufferedImages.get(i).getHeight()][];
            for (int h = 0; h < bufferedImages.get(i).getHeight(); h++) {
                pixels[h] = new BigDecimal[bufferedImages.get(i).getWidth()];
                for (int w = 0; w < bufferedImages.get(i).getWidth(); w++)
                    pixels[h][w] = BigDecimal.valueOf( ((double) new Color(bufferedImages.get(i).getRGB(h, w)).getRed())/255. );
            }
//            pixels.add(1); // смещение

            tmp_trainingPixelsImage.add(pixels);

            pixels = new BigDecimal[bufferedImages.get(i).getHeight()][];
            for (int h = 0; h < bufferedImages.get(i).getHeight(); h++) {
                pixels[h] = new BigDecimal[bufferedImages.get(i).getWidth()];
                for (int w = 0; w < bufferedImages.get(i).getWidth(); w++)
                    pixels[h][w] = BigDecimal.valueOf( ((double) new Color(bufferedImages.get(i).getRGB(h, w)).getGreen())/255. );
            }
//            pixels.add(1); // смещение

            tmp_trainingPixelsImage.add(pixels);

            pixels = new BigDecimal[bufferedImages.get(i).getHeight()][];
            for (int h = 0; h < bufferedImages.get(i).getHeight(); h++) {
                pixels[h] = new BigDecimal[bufferedImages.get(i).getWidth()];
                for (int w = 0; w < bufferedImages.get(i).getWidth(); w++)
                    pixels[h][w] = BigDecimal.valueOf( ((double) new Color(bufferedImages.get(i).getRGB(h, w)).getBlue())/255. );
            }
//            pixels.add(1); // смещение

            tmp_trainingPixelsImage.add(pixels);

            trainingPixelsImages.add(tmp_trainingPixelsImage);
            trainingTrueFalse[bufferedImagesSize/2 - i + bufferedImagesSize - 1] = (byte) 1;

            bufferedImages.get(i).flush();
            bufferedImages.remove(bufferedImages.size() - 1);
        }
        bufferedImages.clear();
        System.out.println("1 bufferedImages0.size: 0");

        NeuralNetwork appleNN = new NeuralNetwork();
        appleNN.training(trainingPixelsImages, trainingTrueFalse);

/*        System.out.println("\n\n\n------------ TESTING -----------\n\n\n");

        folder = new File("not apples/" + 2);
        listFilesForFolder(folder, bufferedImages);
        bufferedImagesSize = bufferedImages.size();
        System.out.println("0 bufferedImages2.size: " + bufferedImagesSize);
        testingTrueFalse = new byte[bufferedImagesSize*2];
        for (int i = bufferedImagesSize - 1; i >= 0; i--) {

            pixels = new BigDecimal[bufferedImages.get(i).getHeight()][];
            for (int h = 0; h < bufferedImages.get(i).getHeight(); h++) {
                pixels[h] = new BigDecimal[bufferedImages.get(i).getWidth()];
                for (int w = 0; w < bufferedImages.get(i).getWidth(); w++)
                    pixels[h][w] = BigDecimal.valueOf( ((double) -bufferedImages.get(i).getRGB(w, h) - 1.)/16777216. );
            }
//            pixels.add(1); // смещение

            testingPixelsImages.add(pixels);
            testingTrueFalse[i] = (byte) 0;

            bufferedImages.get(i).flush();
            bufferedImages.remove(bufferedImages.size() - 1);
        }
        bufferedImages.clear();
        System.out.println("0 bufferedImages2.size: 0");

        folder = new File("apples/" + 2);
        listFilesForFolder(folder, bufferedImages);
        System.out.println("1 bufferedImages2.size: " + bufferedImages.size());
        bufferedImagesSize = bufferedImages.size();
//        testingTrueFalse = new byte[bufferedImagesSize];
        for (int i = bufferedImagesSize - 1; i >= bufferedImagesSize/2; i--) {

            pixels = new BigDecimal[bufferedImages.get(i).getHeight()][];
            for (int h = 0; h < bufferedImages.get(i).getHeight(); h++) {
                pixels[h] = new BigDecimal[bufferedImages.get(i).getWidth()];
                for (int w = 0; w < bufferedImages.get(i).getWidth(); w++)
                    pixels[h][w] = BigDecimal.valueOf( ((double) -bufferedImages.get(i).getRGB(w, h) - 1.)/16777216. );
            }
//            pixels.add(1); // смещение

            testingPixelsImages.add(pixels);
            testingTrueFalse[bufferedImagesSize - 1 + i] = (byte) 1;

            bufferedImages.get(i).flush();
            bufferedImages.remove(bufferedImages.size() - 1);
        }
        bufferedImages.clear();
        System.out.println("1 bufferedImages2.size: 0");

        NeuralNetwork appleTestNN = new NeuralNetwork();
        appleTestNN.testing(testingPixelsImages, testingTrueFalse);*/

        /*for (int f = 0; f < 10; f++) {
            folder = new File("not apples/" + f);
            listFilesForFolder(folder, bufferedImages);
            bufferedImagesSize = bufferedImages.size();
            System.out.println("0 bufferedImages0.size: " + bufferedImagesSize);
            for (int i = bufferedImagesSize - 1; i >= 0; i--) {

                pixels = new ArrayList<>();
                for (int h = 0; h < bufferedImages.get(i).getHeight(); h++) {
                    for (int w = 0; w < bufferedImages.get(i).getWidth(); w++)
                        pixels.add(bufferedImages.get(i).getRGB(w, h));
                }
                pixels.add(1); // смещение

                if (i >= bufferedImagesSize*0.2) { // 80% for training
                    trainingPixelsImages.add(pixels);
                    trainingTrueFalse.add(0);
                }
                else { // 20% for testing
                    testingPixelsImages.add(pixels);
                    testingTrueFalse.add(0);
                }

                bufferedImages.get(i).flush();
                bufferedImages.remove(bufferedImages.size() - 1);
            }
            bufferedImages.clear();
            System.out.println("0 bufferedImages0.size: 0");
        }

        for (int f = 0; f < 10; f++) {
            folder = new File("apples/" + f);
            listFilesForFolder(folder, bufferedImages);
            System.out.println("1 bufferedImages1.size: " + bufferedImages.size());
            bufferedImagesSize = bufferedImages.size();
            for (int i = bufferedImagesSize - 1; i >= 0; i--) {

                pixels = new ArrayList<>();
                for (int h = 0; h < bufferedImages.get(i).getHeight(); h++) {
                    for (int w = 0; w < bufferedImages.get(i).getWidth(); w++)
                        pixels.add(bufferedImages.get(i).getRGB(w, h));
                }
                pixels.add(1); // смещение

                if (i >= bufferedImagesSize*0.2) { // 80% for training
                    trainingPixelsImages.add(pixels);
                    trainingTrueFalse.add(1);
                }
                else { // 20% for testing
                    testingPixelsImages.add(pixels);
                    testingTrueFalse.add(1);
                }

                bufferedImages.get(i).flush();
                bufferedImages.remove(bufferedImages.size() - 1);
            }
            bufferedImages.clear();
            System.out.println("1 bufferedImages1.size: 0");
        }*/

        // очистка памяти
        dimg_resize.flush();
        image_listFilesForFolder.flush();
//        pixels.clear();
        if (!folder.delete())
            System.out.println("Folder is not deleted");

        /*int numberOfThreads = 8;
        Thread[] threads = new Thread[numberOfThreads];
        for (int t = 0; t < numberOfThreads; t++) {
            int finalT = t;
            threads[finalT] = new Thread(() -> { // создание потока
                int start = (int) Math.floor((double) images.size()/numberOfThreads) * finalT;
                int end;
                if (finalT < numberOfThreads - 1)
                    end = (int) Math.floor((double) images.size()/numberOfThreads) * (finalT + 1);
                else
                    end = images.size();
                for (int f = start; f < end; f++) {
                    try {
                        BufferedImage image = ImageIO.read(images.get(f));
                        if ( !(image.getWidth() == 100 && image.getHeight() == 100) )
                            image = resize(image, 100, 100);

//                        ArrayList<int[]> imArray = new ArrayList<>();
//                        int x = (int) Math.round(image.getWidth()/2.);
//                        int y = (int) Math.round(image.getWidth()/2.);
//                        int pixelsColorSum = 0;
//                        for (int xi = x - 1; xi <= x + 1; xi++) {
//                            for (int yi = y - 1; yi <= y + 1; yi++)
//                                pixelsColorSum += Math.round((image.getRGB(xi, yi)));
//                        }
//                        getObjectOnImage(image, x, y, (int) Math.round(pixelsColorSum/9.), imArray);
//
//                        for (int h = 0; h < image.getHeight(); h++) {
//                            outer: for (int w = 0; w < image.getWidth(); w++) {
//                                for (int[] iA : imArray) {
//                                    if (iA[0] == w && iA[1] == h)
//                                        continue outer;
//                                }
//                                image.setRGB(w, h, -1);
//                            }
//                        }
//
//                        image = scaleImage(leftRightImage(image));

                        ArrayList<Integer> im = new ArrayList<>();
                        for (int h = 0; h < image.getHeight(); h++) {
                            for (int w = 0; w < image.getWidth(); w++)
                                im.add(image.getRGB(w, h));
                        }
                        im.add(1); // смещение


//                        // отображение файла
//                        File outputFile = new File("saved.jpg");
//                        BufferedImage newBufferedImage = new BufferedImage(image.getWidth(),
//                                image.getHeight(), BufferedImage.TYPE_INT_RGB);
//                        newBufferedImage.createGraphics().drawImage(image, 0, 0, Color.WHITE, null);
//                        ImageIO.write(newBufferedImage, "jpg", outputFile);
//                        if (!Desktop.isDesktopSupported()) // проверка Desktop поддержки платформой
//                            System.out.println("Desktop is not supported");
//                        else {
//                            Desktop desktop = Desktop.getDesktop();
//                            if(outputFile.exists())
//                                desktop.open(outputFile);
//                        }
//                        System.exit(0);


                        if (f < size)
                            pixelsImagesTrueFalse.add(new PixelsImagesTrueFalse(im, 1));
                        else
                            pixelsImagesTrueFalse.add(new PixelsImagesTrueFalse(im, 0));

                        image.flush();
                    }
                    catch (IOException e) {
                        e.printStackTrace();
                    }
                }

                System.out.println((finalT+1) + " pixelsImages.size: " + pixelsImagesTrueFalse.size());
            });
            threads[finalT].start(); // запуск потока
        }*/

        /*try {
            for (Thread t : threads)
                t.join();
        }
        catch (InterruptedException e) {
            e.printStackTrace();
        }*/

//        NeuralNetwork.training(trainingPixelsImages, trainingTrueFalse);
    }

    public static void listFilesForFolder(final File folder, ArrayList<BufferedImage> bufferedImages) {
        for (final File fileEntry : Objects.requireNonNull(folder.listFiles())) {
            if (fileEntry.isDirectory())
                listFilesForFolder(fileEntry, bufferedImages);
            else {
                try {
                    image_listFilesForFolder = ImageIO.read(fileEntry);
                    if ( !(image_listFilesForFolder.getWidth() == 96 && image_listFilesForFolder.getHeight() == 96) )
                        image_listFilesForFolder = resize(image_listFilesForFolder, 96, 96);

                    bufferedImages.add(image_listFilesForFolder);

                    image_listFilesForFolder.flush();
                }
                catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }

    public static BufferedImage resize(BufferedImage img, int newW, int newH) {
        Image tmp = img.getScaledInstance(newW, newH, Image.SCALE_SMOOTH);
        dimg_resize = new BufferedImage(newW, newH, BufferedImage.TYPE_INT_ARGB);

        Graphics2D g2d = dimg_resize.createGraphics();
        g2d.drawImage(tmp, 0, 0, null);
        g2d.dispose();

        return dimg_resize;
    }

    /*static class NeuralNetwork {

        private static final ArrayList<BigDecimal> synapticWeights = new ArrayList<>();

        public static void training(ArrayList<ArrayList<Integer>> trainingInputs,
                             ArrayList<Integer> trainingOutputs) {

            int numberOfSynaptic = trainingInputs.get(0).size();
            MathContext mathContext20 = new MathContext(20); // для BigDecimal

            for (int i = 0; i < numberOfSynaptic; i++)
                synapticWeights.add(BigDecimal.valueOf(Math.random() - 0.5));

            System.out.println("synapticWeights");

            int n = 20;
            int size = trainingOutputs.size();
            BigDecimal[] output = new BigDecimal[size];
            BigDecimal epsilon = BigDecimal.valueOf(0.6); // скорость обучения (0.7)
            BigDecimal alpha = BigDecimal.valueOf(0.02); // момент (0.3)
            BigDecimal[] deltaWeights = new BigDecimal[numberOfSynaptic];
            Arrays.fill(deltaWeights, BigDecimal.ZERO);
            BigDecimal error;
            BigInteger statistic;
            BigDecimal delta;
            BigDecimal grad;
            double outputDouble;
            for (int i = 0; i < n; i++) {
                output = new BigDecimal[size];
                Arrays.fill(output, BigDecimal.valueOf(0));
                statistic = BigInteger.valueOf(size);

                error = BigDecimal.ZERO;

                for (int j = 0; j < size; j++) {
                    for (int l = 0; l < numberOfSynaptic; l++)
                        output[j] = output[j].add(synapticWeights.get(l).multiply(BigDecimal.valueOf(trainingInputs.get(j).get(l)), mathContext20), mathContext20);

                    outputDouble = sigmoid(output[j].doubleValue());
                    if ((int) Math.round(outputDouble) != trainingOutputs.get(j)) {

                        error = error.add(BigDecimal.valueOf(Math.pow(trainingOutputs.get(j) - outputDouble, 2)/numberOfSynaptic), mathContext20);

                        delta = BigDecimal.valueOf(Math.pow(trainingOutputs.get(j) - outputDouble, 2)/numberOfSynaptic);

                        for (int s = 0; s < numberOfSynaptic; s++) {

                            grad = BigDecimal.valueOf(-trainingInputs.get(j).get(s)).multiply(delta, mathContext20);
                            deltaWeights[s] = epsilon.multiply(grad, mathContext20).add(alpha.multiply(deltaWeights[s], mathContext20), mathContext20);
                            if ((int) Math.round(outputDouble) < trainingOutputs.get(j))
                                synapticWeights.set(s, synapticWeights.get(s).subtract(deltaWeights[s], mathContext20));
                            else
                                synapticWeights.set(s, synapticWeights.get(s).add(deltaWeights[s], mathContext20));
                        }

                        statistic = statistic.subtract(BigInteger.ONE);
                    }
                }

                System.out.println(i + " Error: " + error);

                if (statistic.intValue() == size) {
                    System.out.println(" !!! checkEquals !!!");
                    break;
                }
            }

//            System.out.print("FINAL_OUTPUT: ");

            int stat = 0;

            Arrays.fill(output, BigDecimal.ZERO);
            int intOutput;
            for (int i = 0; i < size; i++) {
                for (int j = 0; j < numberOfSynaptic; j++) {
                    output[i] = output[i].add(BigDecimal.valueOf(trainingInputs.get(i).get(j)).multiply(synapticWeights.get(j), mathContext20), mathContext20);
                }
                intOutput = (int) Math.round(sigmoid(output[i].doubleValue()));
                if (intOutput == trainingOutputs.get(i))
                    stat++;
//                System.out.print(intOutput + " ");
            }
            System.out.println();
            System.out.println("Stat: " + (double) stat/size);

            try (FileWriter fileWriter = new FileWriter("weights.txt", false))
            {
                for (BigDecimal sW : synapticWeights)
                    fileWriter.write(sW.toString() + "\n");

                fileWriter.flush();
            }
            catch(IOException e){
                e.printStackTrace();
            }

            System.out.println("========== END ==========");

            stat = 0;
            size = testingPixelsImages.size();
            ArrayList<Integer> testingOutputs = testing(testingPixelsImages, synapticWeights);
            for (int i = 0; i < size; i++) {
                if (testingOutputs.get(i).equals(testingTrueFalse.get(i)))
                    stat++;
            }
            System.out.println("Stat2: " + (double) stat/size);

            // очистка памяти
            result_testing.clear();
            testingOutputs.clear();
        }

        public static double sigmoid(double x) {
            return 1. / (1. + Math.pow(Math.E, -x));
        }

        public static ArrayList<Integer> testing(ArrayList<ArrayList<Integer>> inputs, ArrayList<BigDecimal> weights) {
            int size = inputs.size();
            int size2 = inputs.get(0).size();
            BigDecimal[] outputs = new BigDecimal[size];
            Arrays.fill(outputs, BigDecimal.ZERO);
            MathContext mathContext20 = new MathContext(20);
            result_testing = new ArrayList<>();
            for (int i = 0; i < size; i++) {
                for (int j = 0; j < size2; j++) {

                    if (inputs.size() == 0)
                        System.out.println("inputs.size() == 0");
                    if (inputs.get(i).size() == 0)
                        System.out.println("inputs.get(i).size() == 0");
                    if (weights.size() == 0)
                        System.out.println("weights.size() == 0");

                    outputs[i] = outputs[i].add(BigDecimal.valueOf(inputs.get(i).get(j)).multiply(weights.get(j), mathContext20), mathContext20);
                }
                result_testing.add( (int) Math.round(sigmoid(outputs[i].doubleValue())) );
            }
            return result_testing;
        }
    }*/

    static class NeuralNetwork {

        private final BigDecimal[][] nucleusWeightsRollUp1_1 = new BigDecimal[5][];
        private final BigDecimal[][] nucleusWeightsRollUp2_1 = new BigDecimal[5][];
        private final BigDecimal[][] nucleusWeightsRollUp1_2 = new BigDecimal[5][];
        private final BigDecimal[][] nucleusWeightsRollUp2_2 = new BigDecimal[5][];
        private final BigDecimal[][] nucleusWeightsRollUp1_3 = new BigDecimal[5][];
        private final BigDecimal[][] nucleusWeightsRollUp2_3 = new BigDecimal[5][];
        private final BigDecimal[][] synapticWeights1_1 = new BigDecimal[21][];
        private final BigDecimal[][] synapticWeights1_2 = new BigDecimal[21][];
        private final BigDecimal[][] synapticWeights1_3 = new BigDecimal[21][];
        private final BigDecimal[] synapticWeights2 = new BigDecimal[3];

        private ArrayList<int[]> maxValueIndexes1_1 = null;
        private ArrayList<int[]> maxValueIndexes2_1 = null;
        private ArrayList<int[]> maxValueIndexes1_2 = null;
        private ArrayList<int[]> maxValueIndexes2_2 = null;
        private ArrayList<int[]> maxValueIndexes1_3 = null;
        private ArrayList<int[]> maxValueIndexes2_3 = null;
        private BigDecimal[][] reduceSampleResult1_1;
        private BigDecimal[][] reduceSampleResult2_1;
        private BigDecimal[][] reduceSampleResult1_2;
        private BigDecimal[][] reduceSampleResult2_2;
        private BigDecimal[][] reduceSampleResult1_3;
        private BigDecimal[][] reduceSampleResult2_3;
        private BigDecimal[] outs1;
        private BigDecimal[] outs2;
        private BigDecimal[] outs3;

        public NeuralNetwork() {
            int length = nucleusWeightsRollUp1_1.length;
            for (int i = 0; i < length; i++) {
                nucleusWeightsRollUp1_1[i] = new BigDecimal[length];
                nucleusWeightsRollUp2_1[i] = new BigDecimal[length];
                nucleusWeightsRollUp1_2[i] = new BigDecimal[length];
                nucleusWeightsRollUp2_2[i] = new BigDecimal[length];
                nucleusWeightsRollUp1_3[i] = new BigDecimal[length];
                nucleusWeightsRollUp2_3[i] = new BigDecimal[length];
                for (int j = 0; j < length; j++) {
                    nucleusWeightsRollUp1_1[i][j] = BigDecimal.valueOf(Math.random() - 0.5);
                    nucleusWeightsRollUp2_1[i][j] = BigDecimal.valueOf(Math.random() - 0.5);
                    nucleusWeightsRollUp1_2[i][j] = BigDecimal.valueOf(Math.random() - 0.5);
                    nucleusWeightsRollUp2_2[i][j] = BigDecimal.valueOf(Math.random() - 0.5);
                    nucleusWeightsRollUp1_3[i][j] = BigDecimal.valueOf(Math.random() - 0.5);
                    nucleusWeightsRollUp2_3[i][j] = BigDecimal.valueOf(Math.random() - 0.5);
                }
            }

            length = synapticWeights1_1.length;
            for (int i = 0; i < length; i++) {
                synapticWeights1_1[i] = new BigDecimal[length];
                synapticWeights1_2[i] = new BigDecimal[length];
                synapticWeights1_3[i] = new BigDecimal[length];
                for (int j = 0; j < length; j++) {
                    synapticWeights1_1[i][j] = BigDecimal.valueOf(Math.random() - 0.5);
                    synapticWeights1_2[i][j] = BigDecimal.valueOf(Math.random() - 0.5);
                    synapticWeights1_3[i][j] = BigDecimal.valueOf(Math.random() - 0.5);
                }
            }

            length = synapticWeights2.length;
            for (int i = 0; i < length; i++)
                synapticWeights2[i] = BigDecimal.valueOf(Math.random() - 0.5);
        }

        public void training(ArrayList<ArrayList<BigDecimal[][]>> trainingInputs, byte[] trainingOutputs) {
            BigDecimal[] outputs = getOutputs(trainingInputs);

            int numberOfEras = 20;                                  // кол-во эпох
            BigDecimal epsilon = BigDecimal.valueOf(0.6);           // скорость обучения (0.7)
            BigDecimal alpha = BigDecimal.valueOf(0.02);            // момент (0.3)
            int numberOfImages = trainingInputs.size();             // кол-во изображений
            final int nucleusLength = synapticWeights1_1.length;    // = 21
            int rollUpNucleusSize = nucleusWeightsRollUp2_1.length; // = 5
            int reduceSample1Size = reduceSampleResult1_1.length;   // = 46

            BigDecimal[] deltaWeightsHO = new BigDecimal[synapticWeights2.length];
            Arrays.fill(deltaWeightsHO, BigDecimal.ZERO);
            BigDecimal[][] deltaWeightsHH1 = new BigDecimal[nucleusLength][];
            Arrays.fill(deltaWeightsHH1, new BigDecimal[nucleusLength]);
            for (int i = 0; i < nucleusLength; i++) {
                for (int j = 0; j < nucleusLength; j++)
                    deltaWeightsHH1[i][j] = BigDecimal.ZERO;
            }
            BigDecimal[][] deltaWeightsHH2 = new BigDecimal[nucleusLength][];
            Arrays.fill(deltaWeightsHH2, new BigDecimal[nucleusLength]);
            for (int i = 0; i < nucleusLength; i++) {
                for (int j = 0; j < nucleusLength; j++)
                    deltaWeightsHH2[i][j] = BigDecimal.ZERO;
            }
            BigDecimal[][] deltaWeightsHH3 = new BigDecimal[nucleusLength][];
            Arrays.fill(deltaWeightsHH3, new BigDecimal[nucleusLength]);
            for (int i = 0; i < nucleusLength; i++) {
                for (int j = 0; j < nucleusLength; j++)
                    deltaWeightsHH3[i][j] = BigDecimal.ZERO;
            }
            BigDecimal[][] deltaWeightsRollUp2_1 = new BigDecimal[rollUpNucleusSize][];
            Arrays.fill(deltaWeightsRollUp2_1, new BigDecimal[rollUpNucleusSize]);
            for (int i = 0; i < rollUpNucleusSize; i++) {
                for (int j = 0; j < rollUpNucleusSize; j++)
                    deltaWeightsRollUp2_1[i][j] = BigDecimal.ZERO;
            }
            BigDecimal[][] deltaWeightsRollUp2_2 = new BigDecimal[rollUpNucleusSize][];
            Arrays.fill(deltaWeightsRollUp2_2, new BigDecimal[rollUpNucleusSize]);
            for (int i = 0; i < rollUpNucleusSize; i++) {
                for (int j = 0; j < rollUpNucleusSize; j++)
                    deltaWeightsRollUp2_2[i][j] = BigDecimal.ZERO;
            }
            BigDecimal[][] deltaWeightsRollUp2_3 = new BigDecimal[rollUpNucleusSize][];
            Arrays.fill(deltaWeightsRollUp2_3, new BigDecimal[rollUpNucleusSize]);
            for (int i = 0; i < rollUpNucleusSize; i++) {
                for (int j = 0; j < rollUpNucleusSize; j++)
                    deltaWeightsRollUp2_3[i][j] = BigDecimal.ZERO;
            }
            BigDecimal[][] deltaWeightsRollUp1_1 = new BigDecimal[rollUpNucleusSize][];
            Arrays.fill(deltaWeightsRollUp1_1, new BigDecimal[rollUpNucleusSize]);
            for (int i = 0; i < rollUpNucleusSize; i++) {
                for (int j = 0; j < rollUpNucleusSize; j++)
                    deltaWeightsRollUp1_1[i][j] = BigDecimal.ZERO;
            }
            BigDecimal[][] deltaWeightsRollUp1_2 = new BigDecimal[rollUpNucleusSize][];
            Arrays.fill(deltaWeightsRollUp1_2, new BigDecimal[rollUpNucleusSize]);
            for (int i = 0; i < rollUpNucleusSize; i++) {
                for (int j = 0; j < rollUpNucleusSize; j++)
                    deltaWeightsRollUp1_2[i][j] = BigDecimal.ZERO;
            }
            BigDecimal[][] deltaWeightsRollUp1_3 = new BigDecimal[rollUpNucleusSize][];
            Arrays.fill(deltaWeightsRollUp1_3, new BigDecimal[rollUpNucleusSize]);
            for (int i = 0; i < rollUpNucleusSize; i++) {
                for (int j = 0; j < rollUpNucleusSize; j++)
                    deltaWeightsRollUp1_3[i][j] = BigDecimal.ZERO;
            }
            BigDecimal[][] deltas_1 = new BigDecimal[nucleusLength][];
            Arrays.fill(deltas_1, new BigDecimal[nucleusLength]);
            BigDecimal[][] deltas_2 = new BigDecimal[nucleusLength][];
            Arrays.fill(deltas_2, new BigDecimal[nucleusLength]);
            BigDecimal[][] deltas_3 = new BigDecimal[nucleusLength][];
            Arrays.fill(deltas_3, new BigDecimal[nucleusLength]);
            BigDecimal deltaO;
            BigDecimal deltaH1;
            BigDecimal deltaH2;
            BigDecimal deltaH3;
            BigDecimal gradH1;
            BigDecimal gradH2;
            BigDecimal gradH3;
            double outputDouble;
            BigDecimal error;
            /*BigInteger statistic;*/
            BigDecimal[][] grads_1;
            BigDecimal[][] grads_2;
            BigDecimal[][] grads_3;
            for (int n = 0; n < numberOfEras; n++) {
                /*statistic = BigInteger.valueOf(size);*/
                error = BigDecimal.ZERO;
                for (int i = 0; i < numberOfImages; i++) {
                    outputDouble = outputs[i].doubleValue();
                    if ( (byte) Math.round(outputDouble) != trainingOutputs[i] ) {

                        error = error.add(BigDecimal.valueOf(Math.pow(trainingOutputs[i] - outputDouble, 2)/2.), mathContext20);

                        // O
                        deltaO = BigDecimal.valueOf(Math.pow((double) trainingOutputs[i] - outputDouble, 2) /synapticWeights2.length); //deltaO = BigDecimal.valueOf(((double) trainingOutputs[i] - outputDouble)*(1. - outputDouble)*outputDouble);

                        // H1, H2, H3
                        deltaH1 = (BigDecimal.ONE.subtract(outs1[i], mathContext20)).multiply(outs1[i], mathContext20).multiply(deltaO, mathContext20).multiply(synapticWeights2[0], mathContext20);
                        gradH1 = deltaO.multiply(outs1[i], mathContext20);
                        deltaWeightsHO[0] = epsilon.multiply(gradH1, mathContext20).add(alpha.multiply(deltaWeightsHO[0], mathContext20), mathContext20);
                        synapticWeights2[0] = synapticWeights2[0].add(deltaWeightsHO[0], mathContext20);

                        deltaH2 = (BigDecimal.ONE.subtract(outs2[i], mathContext20)).multiply(outs2[i], mathContext20).multiply(deltaO, mathContext20).multiply(synapticWeights2[1], mathContext20);
                        gradH2 = deltaO.multiply(outs2[i], mathContext20);
                        deltaWeightsHO[1] = epsilon.multiply(gradH2, mathContext20).add(alpha.multiply(deltaWeightsHO[1], mathContext20), mathContext20);
                        synapticWeights2[1] = synapticWeights2[1].add(deltaWeightsHO[1], mathContext20);

                        deltaH3 = (BigDecimal.ONE.subtract(outs3[i], mathContext20)).multiply(outs3[i], mathContext20).multiply(deltaO, mathContext20).multiply(synapticWeights2[2], mathContext20);
                        gradH3 = deltaO.multiply(outs3[i], mathContext20);
                        deltaWeightsHO[2] = epsilon.multiply(gradH3, mathContext20).add(alpha.multiply(deltaWeightsHO[2], mathContext20), mathContext20);
                        synapticWeights2[2] = synapticWeights2[2].add(deltaWeightsHO[2], mathContext20);

                        // подвыборочный слой 2
                        for (int h = 0; h < nucleusLength; h++) { // nucleusLength = 21
                            for (int w = 0; w < nucleusLength; w++) {
                                deltas_1[h][w] = (BigDecimal.ONE.subtract(reduceSampleResult2_1[h][w], mathContext20)).multiply(reduceSampleResult2_1[h][w], mathContext20).multiply(deltaH1, mathContext20).multiply(synapticWeights1_1[h][w], mathContext20);
                                gradH1 = deltaH1.multiply(reduceSampleResult2_1[h][w], mathContext20);
                                deltaWeightsHH1[h][w] = epsilon.multiply(gradH1, mathContext20).add(alpha.multiply(deltaWeightsHH1[h][w], mathContext20), mathContext20);
                                synapticWeights1_1[h][w] = synapticWeights1_1[h][w].add(deltaWeightsHH1[h][w], mathContext20);

                                deltas_2[h][w] = (BigDecimal.ONE.subtract(reduceSampleResult2_2[h][w], mathContext20)).multiply(reduceSampleResult2_2[h][w], mathContext20).multiply(deltaH2, mathContext20).multiply(synapticWeights1_2[h][w], mathContext20);
                                gradH2 = deltaH2.multiply(reduceSampleResult2_2[h][w], mathContext20);
                                deltaWeightsHH2[h][w] = epsilon.multiply(gradH2, mathContext20).add(alpha.multiply(deltaWeightsHH2[h][w], mathContext20), mathContext20);
                                synapticWeights1_2[h][w] = synapticWeights1_2[h][w].add(deltaWeightsHH2[h][w], mathContext20);

                                deltas_3[h][w] = (BigDecimal.ONE.subtract(reduceSampleResult2_3[h][w], mathContext20)).multiply(reduceSampleResult2_3[h][w], mathContext20).multiply(deltaH3, mathContext20).multiply(synapticWeights1_3[h][w], mathContext20);
                                gradH3 = deltaH3.multiply(reduceSampleResult2_3[h][w], mathContext20);
                                deltaWeightsHH3[h][w] = epsilon.multiply(gradH3, mathContext20).add(alpha.multiply(deltaWeightsHH3[h][w], mathContext20), mathContext20);
                                synapticWeights1_3[h][w] = synapticWeights1_3[h][w].add(deltaWeightsHH3[h][w], mathContext20);
                            }
                        }

                        // сверточный слой 2
                        BigDecimal[][] deltaMatrix2_1 = new BigDecimal[nucleusLength*2][];
                        Arrays.fill(deltaMatrix2_1, new BigDecimal[nucleusLength*2]);
                        for (int h = 0; h < nucleusLength*2; h++)
                            Arrays.fill(deltaMatrix2_1[h], BigDecimal.ZERO);
                        int x = 0;
                        int length = reduceSampleResult2_1.length; // = nucleusLength = 21
                        for (int h = 0; h < length; h++) {
                            for (int w = 0; w < length; w++) {
                                if (!reduceSampleResult2_1[h][w].equals(BigDecimal.ZERO)) {
                                    deltaMatrix2_1[(maxValueIndexes2_1.get(x))[0]][(maxValueIndexes2_1.get(x))[1]] = deltas_1[h][w]; //synapticWeights1_1[h][w];
                                    x++;
                                }
                            }
                        }

                        BigDecimal[][] deltaMatrix2_2 = new BigDecimal[nucleusLength*2][];
                        Arrays.fill(deltaMatrix2_2, new BigDecimal[nucleusLength*2]);
                        for (int h = 0; h < nucleusLength*2; h++)
                            Arrays.fill(deltaMatrix2_2[h], BigDecimal.ZERO);
                        x = 0;
                        for (int h = 0; h < length; h++) {
                            for (int w = 0; w < length; w++) {
                                if (!reduceSampleResult2_2[h][w].equals(BigDecimal.ZERO)) {
                                    deltaMatrix2_2[(maxValueIndexes2_2.get(x))[0]][(maxValueIndexes2_2.get(x))[1]] = deltas_2[h][w]; //synapticWeights1_2[h][w];
                                    x++;
                                }
                            }
                        }

                        BigDecimal[][] deltaMatrix2_3 = new BigDecimal[nucleusLength*2][];
                        Arrays.fill(deltaMatrix2_3, new BigDecimal[nucleusLength*2]);
                        for (int h = 0; h < nucleusLength*2; h++)
                            Arrays.fill(deltaMatrix2_3[h], BigDecimal.ZERO);
                        x = 0;
                        for (int h = 0; h < length; h++) {
                            for (int w = 0; w < length; w++) {
                                if (!reduceSampleResult2_3[h][w].equals(BigDecimal.ZERO)) {
                                    deltaMatrix2_3[(maxValueIndexes2_3.get(x))[0]][(maxValueIndexes2_3.get(x))[1]] = deltas_3[h][w]; //synapticWeights1_3[h][w];
                                    x++;
                                }
                            }
                        }

                        // подвыборочный слой 1
                        deltas_1 = reverseConvolution(deltaMatrix2_1, nucleusWeightsRollUp2_1);
                        deltas_2 = reverseConvolution(deltaMatrix2_2, nucleusWeightsRollUp2_2);
                        deltas_3 = reverseConvolution(deltaMatrix2_3, nucleusWeightsRollUp2_3);
                        grads_1 = rollUp(reduceSampleResult1_1, deltaMatrix2_1);
                        grads_2 = rollUp(reduceSampleResult1_2, deltaMatrix2_2);
                        grads_3 = rollUp(reduceSampleResult1_3, deltaMatrix2_3);
                        for (int h = 0; h < rollUpNucleusSize; h++) { // length = 5
                            for (int w = 0; w < rollUpNucleusSize; w++) { // length = 5
                                deltaWeightsRollUp2_1[h][w] = epsilon.multiply(grads_1[h][w], mathContext20).add(alpha.multiply(deltaWeightsRollUp2_1[h][w], mathContext20), mathContext20);
                                deltaWeightsRollUp2_2[h][w] = epsilon.multiply(grads_2[h][w], mathContext20).add(alpha.multiply(deltaWeightsRollUp2_2[h][w], mathContext20), mathContext20);
                                deltaWeightsRollUp2_3[h][w] = epsilon.multiply(grads_3[h][w], mathContext20).add(alpha.multiply(deltaWeightsRollUp2_3[h][w], mathContext20), mathContext20);
                                nucleusWeightsRollUp2_1[h][w] = nucleusWeightsRollUp2_1[h][w].add(deltaWeightsRollUp2_1[h][w], mathContext20);
                                nucleusWeightsRollUp2_2[h][w] = nucleusWeightsRollUp2_2[h][w].add(deltaWeightsRollUp2_2[h][w], mathContext20);
                                nucleusWeightsRollUp2_3[h][w] = nucleusWeightsRollUp2_3[h][w].add(deltaWeightsRollUp2_3[h][w], mathContext20);
                            }
                        }

                        // сверточный слой 1
                        BigDecimal[][] deltaMatrix1_1 = new BigDecimal[reduceSample1Size*2][];
                        Arrays.fill(deltaMatrix1_1, new BigDecimal[reduceSample1Size*2]);
                        for (int h = 0; h < reduceSample1Size*2; h++)
                            Arrays.fill(deltaMatrix1_1[h], BigDecimal.ZERO);
                        x = 0;
                        for (int h = 0; h < reduceSample1Size; h++) {
                            for (int w = 0; w < reduceSample1Size; w++) {
                                if (!reduceSampleResult1_1[h][w].equals(BigDecimal.ZERO)) {
                                    deltaMatrix1_1[(maxValueIndexes1_1.get(x))[0]][(maxValueIndexes1_1.get(x))[1]] = deltas_1[h][w]; // nucleusWeightsRollUp2_1[h][w];
                                    x++;
                                }
                            }
                        }

                        BigDecimal[][] deltaMatrix1_2 = new BigDecimal[reduceSample1Size*2][];
                        Arrays.fill(deltaMatrix1_2, new BigDecimal[reduceSample1Size*2]);
                        for (int h = 0; h < reduceSample1Size*2; h++)
                            Arrays.fill(deltaMatrix1_2[h], BigDecimal.ZERO);
                        x = 0;
                        for (int h = 0; h < reduceSample1Size; h++) {
                            for (int w = 0; w < reduceSample1Size; w++) {
                                if (!reduceSampleResult1_2[h][w].equals(BigDecimal.ZERO)) {
                                    deltaMatrix1_2[(maxValueIndexes1_2.get(x))[0]][(maxValueIndexes1_2.get(x))[1]] = deltas_2[h][w]; // nucleusWeightsRollUp2_2[h][w];
                                    x++;
                                }
                            }
                        }

                        BigDecimal[][] deltaMatrix1_3 = new BigDecimal[reduceSample1Size*2][];
                        Arrays.fill(deltaMatrix1_3, new BigDecimal[reduceSample1Size*2]);
                        for (int h = 0; h < reduceSample1Size*2; h++)
                            Arrays.fill(deltaMatrix1_3[h], BigDecimal.ZERO);
                        x = 0;
                        for (int h = 0; h < reduceSample1Size; h++) {
                            for (int w = 0; w < reduceSample1Size; w++) {
                                if (!reduceSampleResult1_3[h][w].equals(BigDecimal.ZERO)) {
                                    deltaMatrix1_3[(maxValueIndexes1_3.get(x))[0]][(maxValueIndexes1_3.get(x))[1]] = deltas_3[h][w]; // nucleusWeightsRollUp2_2[h][w];
                                    x++;
                                }
                            }
                        }

                        // входной слой
                        grads_1 = rollUp(trainingInputs.get(i).get(0), deltaMatrix1_1);
                        grads_2 = rollUp(trainingInputs.get(i).get(1), deltaMatrix1_2);
                        grads_3 = rollUp(trainingInputs.get(i).get(2), deltaMatrix1_3);
                        for (int h = 0; h < rollUpNucleusSize; h++) { // length = 5
                            for (int w = 0; w < rollUpNucleusSize; w++) { // length = 5
                                deltaWeightsRollUp1_1[h][w] = epsilon.multiply(grads_1[h][w], mathContext20).add(alpha.multiply(deltaWeightsRollUp1_1[h][w], mathContext20), mathContext20);
                                deltaWeightsRollUp1_2[h][w] = epsilon.multiply(grads_2[h][w], mathContext20).add(alpha.multiply(deltaWeightsRollUp1_2[h][w], mathContext20), mathContext20);
                                deltaWeightsRollUp1_3[h][w] = epsilon.multiply(grads_3[h][w], mathContext20).add(alpha.multiply(deltaWeightsRollUp1_3[h][w], mathContext20), mathContext20);
                                nucleusWeightsRollUp1_1[h][w] = nucleusWeightsRollUp1_1[h][w].add(deltaWeightsRollUp1_1[h][w], mathContext20);
                                nucleusWeightsRollUp1_2[h][w] = nucleusWeightsRollUp1_2[h][w].add(deltaWeightsRollUp1_2[h][w], mathContext20);
                                nucleusWeightsRollUp1_3[h][w] = nucleusWeightsRollUp1_3[h][w].add(deltaWeightsRollUp1_3[h][w], mathContext20);
                            }
                        }


                        /*statistic = BigInteger.valueOf(size);*/

                        // очистка памяти
                        maxValueIndexes1_1 = null;
                        maxValueIndexes2_1 = null;
                        maxValueIndexes1_2 = null;
                        maxValueIndexes2_2 = null;
                    }
                }

                System.out.println(n + " Error: " + error);

                /*if (statistic.intValue() == size) {
                    System.out.println(" !!! checkEquals !!!");
                    break;
                }*/
            }


            outputs = getOutputs(trainingInputs);
            int stat = 0;
            for (int i = 0; i < numberOfImages; i++) {
                if ((byte) Math.round(outputs[i].doubleValue()) == trainingOutputs[i])
                    stat++;
            }
            System.out.println("Stat: " + (double) stat/numberOfImages);

            writeWeightsToFile();

            System.out.println("========== END ==========");

            /*testing(testingPixelsImages, testingTrueFalse);*/
        }

        // для сверточных слоев
        public BigDecimal[][] rollUp(BigDecimal[][] input, BigDecimal[][] nucleusWeights) {
            int nucleusSize = nucleusWeights.length;
            int convMapSize = input.length - nucleusSize + 1;

            BigDecimal[][] convolutionalMap = new BigDecimal[convMapSize][];
            Arrays.fill(convolutionalMap, new BigDecimal[convMapSize]);
            for (BigDecimal[] cM : convolutionalMap)
                Arrays.fill(cM, BigDecimal.ZERO);

            BigDecimal sumValue;
            for (int i = 0; i < convMapSize; i++) {
                for (int j = 0; j < convMapSize; j++) {
                    sumValue = BigDecimal.ZERO;
                    for (int ii = 0; ii < nucleusSize; ii++) {
                        for (int jj = 0; jj < nucleusSize; jj++)
                            sumValue = sumValue.add(input[i + ii][j + jj].multiply(nucleusWeights[ii][jj], mathContext20), mathContext20);
                    }
                    convolutionalMap[i][j] = (sumValue.compareTo(BigDecimal.ZERO) > 0) ? sumValue : BigDecimal.ZERO;
                }
            }

            return convolutionalMap;
        }

        // для подвыборочных слоев
        public BigDecimal[][] reduceSample(BigDecimal[][] input) {
            int nucleusSize = 2;
            int subSampleMapSize = input.length/2;

            BigDecimal[][] subSampleMap = new BigDecimal[subSampleMapSize][];
            Arrays.fill(subSampleMap, new BigDecimal[subSampleMapSize]);
            for (BigDecimal[] sM : subSampleMap)
                Arrays.fill(sM, BigDecimal.ZERO);

            if (maxValueIndexes1_1 == null) {
                maxValueIndexes1_1 = new ArrayList<>();
                int[] maxValueIndex;
                BigDecimal maxValue;
                for (int i = 0; i < subSampleMapSize; i += nucleusSize) {
                    for (int j = 0; j < subSampleMapSize; j += nucleusSize) {
                        maxValue = BigDecimal.ZERO;
                        maxValueIndex = new int[] {-1, -1};
                        for (int ii = 0; ii < nucleusSize; ii++) {
                            for (int jj = 0; jj < nucleusSize; jj++) {
                                if (maxValue.compareTo(input[i*nucleusSize + ii][j*nucleusSize + jj]) < 0) {
                                    maxValue = input[i*nucleusSize + ii][j*nucleusSize + jj];
                                    maxValueIndex[0] = i*nucleusSize + ii;
                                    maxValueIndex[1] = j*nucleusSize + jj;
                                }
                            }
                        }
                        subSampleMap[i][j] = maxValue;
                        if (!maxValue.equals(BigDecimal.ZERO))
                            maxValueIndexes1_1.add(maxValueIndex);
                    }
                }
            }
            else if (maxValueIndexes2_1 == null) {
                maxValueIndexes2_1 = new ArrayList<>();
                int[] maxValueIndex;
                BigDecimal maxValue;
                for (int i = 0; i < subSampleMapSize; i += nucleusSize) {
                    for (int j = 0; j < subSampleMapSize; j += nucleusSize) {
                        maxValue = BigDecimal.ZERO;
                        maxValueIndex = new int[] {-1, -1};
                        for (int ii = 0; ii < nucleusSize; ii++) {
                            for (int jj = 0; jj < nucleusSize; jj++) {
                                if (maxValue.compareTo(input[i*nucleusSize + ii][j*nucleusSize + jj]) < 0) {
                                    maxValue = input[i*nucleusSize + ii][j*nucleusSize + jj];
                                    maxValueIndex[0] = i*nucleusSize + ii;
                                    maxValueIndex[1] = j*nucleusSize + jj;
                                }
                            }
                        }
                        subSampleMap[i][j] = maxValue;
                        if (!maxValue.equals(BigDecimal.ZERO))
                            maxValueIndexes2_1.add(maxValueIndex);
                    }
                }
            }
            else if (maxValueIndexes1_2 == null) {
                maxValueIndexes1_2 = new ArrayList<>();
                int[] maxValueIndex;
                BigDecimal maxValue;
                for (int i = 0; i < subSampleMapSize; i += nucleusSize) {
                    for (int j = 0; j < subSampleMapSize; j += nucleusSize) {
                        maxValue = BigDecimal.ZERO;
                        maxValueIndex = new int[] {-1, -1};
                        for (int ii = 0; ii < nucleusSize; ii++) {
                            for (int jj = 0; jj < nucleusSize; jj++) {
                                if (maxValue.compareTo(input[i*nucleusSize + ii][j*nucleusSize + jj]) < 0) {
                                    maxValue = input[i*nucleusSize + ii][j*nucleusSize + jj];
                                    maxValueIndex[0] = i*nucleusSize + ii;
                                    maxValueIndex[1] = j*nucleusSize + jj;
                                }
                            }
                        }
                        subSampleMap[i][j] = maxValue;
                        if (!maxValue.equals(BigDecimal.ZERO))
                            maxValueIndexes1_2.add(maxValueIndex);
                    }
                }
            }
            else if (maxValueIndexes2_2 == null) {
                maxValueIndexes2_2 = new ArrayList<>();
                int[] maxValueIndex;
                BigDecimal maxValue;
                for (int i = 0; i < subSampleMapSize; i += nucleusSize) {
                    for (int j = 0; j < subSampleMapSize; j += nucleusSize) {
                        maxValue = BigDecimal.ZERO;
                        maxValueIndex = new int[] {-1, -1};
                        for (int ii = 0; ii < nucleusSize; ii++) {
                            for (int jj = 0; jj < nucleusSize; jj++) {
                                if (maxValue.compareTo(input[i*nucleusSize + ii][j*nucleusSize + jj]) < 0) {
                                    maxValue = input[i*nucleusSize + ii][j*nucleusSize + jj];
                                    maxValueIndex[0] = i*nucleusSize + ii;
                                    maxValueIndex[1] = j*nucleusSize + jj;
                                }
                            }
                        }
                        subSampleMap[i][j] = maxValue;
                        if (!maxValue.equals(BigDecimal.ZERO))
                            maxValueIndexes2_2.add(maxValueIndex);
                    }
                }
            }
            else if (maxValueIndexes1_3 == null) {
                maxValueIndexes1_3 = new ArrayList<>();
                int[] maxValueIndex;
                BigDecimal maxValue;
                for (int i = 0; i < subSampleMapSize; i += nucleusSize) {
                    for (int j = 0; j < subSampleMapSize; j += nucleusSize) {
                        maxValue = BigDecimal.ZERO;
                        maxValueIndex = new int[] {-1, -1};
                        for (int ii = 0; ii < nucleusSize; ii++) {
                            for (int jj = 0; jj < nucleusSize; jj++) {
                                if (maxValue.compareTo(input[i*nucleusSize + ii][j*nucleusSize + jj]) < 0) {
                                    maxValue = input[i*nucleusSize + ii][j*nucleusSize + jj];
                                    maxValueIndex[0] = i*nucleusSize + ii;
                                    maxValueIndex[1] = j*nucleusSize + jj;
                                }
                            }
                        }
                        subSampleMap[i][j] = maxValue;
                        if (!maxValue.equals(BigDecimal.ZERO))
                            maxValueIndexes1_3.add(maxValueIndex);
                    }
                }
            }
            else if (maxValueIndexes2_3 == null) {
                maxValueIndexes2_3 = new ArrayList<>();
                int[] maxValueIndex;
                BigDecimal maxValue;
                for (int i = 0; i < subSampleMapSize; i += nucleusSize) {
                    for (int j = 0; j < subSampleMapSize; j += nucleusSize) {
                        maxValue = BigDecimal.ZERO;
                        maxValueIndex = new int[] {-1, -1};
                        for (int ii = 0; ii < nucleusSize; ii++) {
                            for (int jj = 0; jj < nucleusSize; jj++) {
                                if (maxValue.compareTo(input[i*nucleusSize + ii][j*nucleusSize + jj]) < 0) {
                                    maxValue = input[i*nucleusSize + ii][j*nucleusSize + jj];
                                    maxValueIndex[0] = i*nucleusSize + ii;
                                    maxValueIndex[1] = j*nucleusSize + jj;
                                }
                            }
                        }
                        subSampleMap[i][j] = maxValue;
                        if (!maxValue.equals(BigDecimal.ZERO))
                            maxValueIndexes2_3.add(maxValueIndex);
                    }
                }
            }

            return subSampleMap;
        }

        // для полносвязных слоев
        public BigDecimal multiplyByWeightsAndAdd(BigDecimal[][] input, BigDecimal[][] weights) {
            BigDecimal output = BigDecimal.ZERO;
            int length = input.length;
            for (int i = 0; i < length; i++) {
                for (int j = 0; j < length; j++)
                    output = output.add(input[i][j].multiply(weights[i][j], mathContext20), mathContext20);
            }
            return output;
        }
        public BigDecimal multiplyByWeightsAndAdd(BigDecimal[] inputs, BigDecimal[] weights) {
            BigDecimal output = BigDecimal.ZERO;
            for (int i = 0; i < inputs.length; i++)
                output = output.add(inputs[i].multiply(weights[i], mathContext20), mathContext20);
            return output;
        }

        public double sigmoid(double x) {
            return 1. / (1. + Math.pow(Math.E, -x));
        }

        public BigDecimal[] getOutputs(ArrayList<ArrayList<BigDecimal[][]>> inputs) {
            int size = inputs.size();
            outs1 = new BigDecimal[size];
            outs2 = new BigDecimal[size];
            outs3 = new BigDecimal[size];
            BigDecimal[] outputs = new BigDecimal[size];
            for (int i = 0; i < size; i++) {
                reduceSampleResult1_1 = reduceSample(rollUp(inputs.get(i).get(0), nucleusWeightsRollUp1_1));
                reduceSampleResult2_1 = reduceSample(rollUp(reduceSampleResult1_1, nucleusWeightsRollUp2_1));
                reduceSampleResult1_2 = reduceSample(rollUp(inputs.get(i).get(1), nucleusWeightsRollUp1_2));
                reduceSampleResult2_2 = reduceSample(rollUp(reduceSampleResult1_2, nucleusWeightsRollUp2_2));
                reduceSampleResult1_3 = reduceSample(rollUp(inputs.get(i).get(2), nucleusWeightsRollUp1_3));
                reduceSampleResult2_3 = reduceSample(rollUp(reduceSampleResult1_3, nucleusWeightsRollUp2_3));
                outs1[i] = BigDecimal.valueOf(sigmoid(multiplyByWeightsAndAdd(reduceSampleResult2_1, synapticWeights1_1).doubleValue()));
                outs2[i] = BigDecimal.valueOf(sigmoid(multiplyByWeightsAndAdd(reduceSampleResult2_2, synapticWeights1_2).doubleValue()));
                outs3[i] = BigDecimal.valueOf(sigmoid(multiplyByWeightsAndAdd(reduceSampleResult2_3, synapticWeights1_3).doubleValue()));
                outputs[i] = BigDecimal.valueOf(sigmoid(multiplyByWeightsAndAdd(new BigDecimal[] {outs1[i], outs2[i], outs3[i]}, synapticWeights2).doubleValue()));
            }
            return outputs;
        }

        public void testing(ArrayList<ArrayList<BigDecimal[][]>> testingInputs, byte[] testingOutputs) {
            BigDecimal[] outputs = getOutputs(testingInputs);
            int stat = 0;
            int size = testingInputs.size();
            for (int i = 0; i < size; i++) {
                if ((byte) Math.round(outputs[i].doubleValue()) == testingOutputs[i])
                    stat++;
            }
            System.out.println("StatTesting: " + (double) stat/size);
        }

        public void writeWeightsToFile() {
            try (FileWriter fileWriter = new FileWriter("weights.txt", false))
            {
                for (BigDecimal sW2 : synapticWeights2)
                    fileWriter.write(sW2.toString() + " ");
                fileWriter.write("\n\n\n");

                for (BigDecimal[] sW11 : synapticWeights1_1) {
                    for (BigDecimal s : sW11)
                        fileWriter.write(s.toString() + " ");
                    fileWriter.write("\n");
                }
                fileWriter.write("\n");
                for (BigDecimal[] sW12 : synapticWeights1_2) {
                    for (BigDecimal s : sW12)
                        fileWriter.write(s.toString() + " ");
                    fileWriter.write("\n");
                }
                fileWriter.write("\n");
                for (BigDecimal[] sW13 : synapticWeights1_3) {
                    for (BigDecimal s : sW13)
                        fileWriter.write(s.toString() + " ");
                    fileWriter.write("\n");
                }
                fileWriter.write("\n\n");

                for (BigDecimal[] nWR21 : nucleusWeightsRollUp2_1) {
                    for (BigDecimal n : nWR21)
                        fileWriter.write(n.toString() + " ");
                    fileWriter.write("\n");
                }
                fileWriter.write("\n");
                for (BigDecimal[] nWR22 : nucleusWeightsRollUp2_2) {
                    for (BigDecimal n : nWR22)
                        fileWriter.write(n.toString() + " ");
                    fileWriter.write("\n");
                }
                fileWriter.write("\n");
                for (BigDecimal[] nWR23 : nucleusWeightsRollUp2_3) {
                    for (BigDecimal n : nWR23)
                        fileWriter.write(n.toString() + " ");
                    fileWriter.write("\n");
                }
                fileWriter.write("\n\n");

                for (BigDecimal[] nWR11 : nucleusWeightsRollUp1_1) {
                    for (BigDecimal n : nWR11)
                        fileWriter.write(n.toString() + " ");
                    fileWriter.write("\n");
                }
                fileWriter.write("\n");
                for (BigDecimal[] nWR12 : nucleusWeightsRollUp1_2) {
                    for (BigDecimal n : nWR12)
                        fileWriter.write(n.toString() + " ");
                    fileWriter.write("\n");
                }
                fileWriter.write("\n");
                for (BigDecimal[] nWR13 : nucleusWeightsRollUp1_3) {
                    for (BigDecimal n : nWR13)
                        fileWriter.write(n.toString() + " ");
                    fileWriter.write("\n");
                }

                fileWriter.flush();
            }
            catch(IOException e){
                e.printStackTrace();
            }
        }

        // обратная свертка
        public BigDecimal[][] reverseConvolution(BigDecimal[][] deltas, BigDecimal[][] nucleus) {
            int deltasLength = deltas.length;
            int nucleusLength = nucleus.length;

            BigDecimal[][] deltasWithBorders = new BigDecimal[deltasLength + nucleusLength - 1][];
            Arrays.fill(deltasWithBorders, new BigDecimal[deltasLength + nucleusLength - 1]);
            for (int i = 0; i < deltasWithBorders.length; i++) {
                for (int j = 0; j < deltasWithBorders.length; j++) {
                    if ( i < (int) Math.floor(nucleusLength/2.)
                            || i > deltasWithBorders.length - 1 - (int) Math.floor(nucleusLength/2.)
                            || j < (int) Math.floor(nucleusLength/2.)
                            || j > deltasWithBorders.length - 1 - (int) Math.floor(nucleusLength/2.) )
                        deltasWithBorders[i][j] = BigDecimal.ZERO;
                    else
                        deltasWithBorders[i][j] = deltas[i - (int) Math.floor(nucleusLength/2.)][j - (int) Math.floor(nucleusLength/2.)];
                }
            }

            /*BigDecimal[][] rotateNucleus = new BigDecimal[nucleusLength][];
            Arrays.fill(rotateNucleus, new BigDecimal[nucleusLength]);
            for (int i = 0; i < nucleusLength; i++) {
                for (int j = 0; j < nucleusLength; j++)
                    rotateNucleus[i][j] = nucleus[nucleusLength - 1 - i][nucleusLength - 1 - j];
            }

            return rollUp(deltasWithBorders, rotateNucleus);*/
            return rollUp(deltasWithBorders, nucleus);
        }
    }
}
