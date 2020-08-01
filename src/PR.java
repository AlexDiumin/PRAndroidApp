import org.opencv.core.Core;
import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.math.BigDecimal;
import java.math.BigInteger;
import java.math.MathContext;
import java.util.*;


public class PR {

    static { System.loadLibrary(Core.NATIVE_LIBRARY_NAME); }

    private static final ArrayList<BufferedImage> bufferedImages = new ArrayList<>();

    private static final ArrayList<ArrayList<Integer>> trainingPixelsImages = new ArrayList<>();
    private static final ArrayList<Integer> trainingTrueFalse = new ArrayList<>();

    private static final ArrayList<ArrayList<Integer>> testingPixelsImages = new ArrayList<>();
    private static final ArrayList<Integer> testingTrueFalse = new ArrayList<>();

    private static BufferedImage dimg_resize;
    private static BufferedImage image_listFilesForFolder;
    private static ArrayList<Integer> result_testing = new ArrayList<>();

    public static void main(String[] args) {

        int bufferedImagesSize;
        ArrayList<Integer> pixels;
        File folder = null;

        for (int f = 0; f < 10; f++) {
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
        }

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

        NeuralNetwork.training(trainingPixelsImages, trainingTrueFalse);
    }

    public static void listFilesForFolder(final File folder, ArrayList<BufferedImage> bufferedImages) {
        for (final File fileEntry : Objects.requireNonNull(folder.listFiles())) {
            if (fileEntry.isDirectory())
                listFilesForFolder(fileEntry, bufferedImages);
            else {
                try {
                    image_listFilesForFolder = ImageIO.read(fileEntry);
                    if ( !(image_listFilesForFolder.getWidth() == 50 && image_listFilesForFolder.getHeight() == 50) )
                        image_listFilesForFolder = resize(image_listFilesForFolder, 50, 50);

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

    static class NeuralNetwork {

        private static final ArrayList<BigDecimal> synapticWeights = new ArrayList<>();

        public static void training(ArrayList<ArrayList<Integer>> trainingInputs,
                             ArrayList<Integer> trainingOutputs) {

            int numberOfSynaptic = trainingInputs.get(0).size();
            MathContext mathContext20 = new MathContext(20); // для BigDecimal

            for (int i = 0; i < numberOfSynaptic; i++)
                synapticWeights.add(BigDecimal.valueOf(2 * Math.random() - 1));

            System.out.println("synapticWeights");

            int n = 20;
            int size = trainingOutputs.size();
            BigDecimal[] output = new BigDecimal[size];
            BigDecimal epsilon = BigDecimal.valueOf(0.7); // скорость обучения (0.7)
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
    }

    /*static class PixelsImagesTrueFalse {
        private final ArrayList<Integer> pixelsImages;
        private final int trueFalse;

        public PixelsImagesTrueFalse(ArrayList<Integer> pixelsImages, int trueFalse) {
            this.pixelsImages = pixelsImages;
            this.trueFalse = trueFalse;
        }
    }*/
}
