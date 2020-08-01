import org.opencv.core.Core;
import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.geom.AffineTransform;
import java.awt.image.AffineTransformOp;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.math.BigDecimal;
import java.math.BigInteger;
import java.math.MathContext;
import java.util.*;


public class PR {

    static { System.loadLibrary(Core.NATIVE_LIBRARY_NAME); }

    private static final ArrayList<File> images = new ArrayList<>();
    private static final ArrayList<PixelsImagesTrueFalse> pixelsImagesTrueFalse = new ArrayList<>();

    public static void main(String[] args) {

        int size;
        final File folder1 = new File("apples SMALL");
        listFilesForFolder(folder1);
        size = images.size();
        System.out.println("01 images.size: " + images.size());
        final File folder2 = new File("not apples SMALL");
        listFilesForFolder(folder2);
        System.out.println("02 images.size: " + images.size());

        int numberOfThreads = 8;
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

                        /*ArrayList<int[]> imArray = new ArrayList<>();
                        int x = (int) Math.round(image.getWidth()/2.);
                        int y = (int) Math.round(image.getWidth()/2.);
                        int pixelsColorSum = 0;
                        for (int xi = x - 1; xi <= x + 1; xi++) {
                            for (int yi = y - 1; yi <= y + 1; yi++)
                                pixelsColorSum += Math.round((image.getRGB(xi, yi)));
                        }
                        getObjectOnImage(image, x, y, (int) Math.round(pixelsColorSum/9.), imArray);

                        for (int h = 0; h < image.getHeight(); h++) {
                            outer: for (int w = 0; w < image.getWidth(); w++) {
                                for (int[] iA : imArray) {
                                    if (iA[0] == w && iA[1] == h)
                                        continue outer;
                                }
                                image.setRGB(w, h, -1);
                            }
                        }

                        image = scaleImage(leftRightImage(image));*/

                        ArrayList<Integer> im = new ArrayList<>();
                        for (int h = 0; h < image.getHeight(); h++) {
                            for (int w = 0; w < image.getWidth(); w++)
                                im.add(image.getRGB(w, h));
                        }
                        im.add(1); // смещение


                        /*// отображение файла
                        File outputFile = new File("saved.jpg");
                        BufferedImage newBufferedImage = new BufferedImage(image.getWidth(),
                                image.getHeight(), BufferedImage.TYPE_INT_RGB);
                        newBufferedImage.createGraphics().drawImage(image, 0, 0, Color.WHITE, null);
                        ImageIO.write(newBufferedImage, "jpg", outputFile);
                        if (!Desktop.isDesktopSupported()) // проверка Desktop поддержки платформой
                            System.out.println("Desktop is not supported");
                        else {
                            Desktop desktop = Desktop.getDesktop();
                            if(outputFile.exists())
                                desktop.open(outputFile);
                        }
                        System.exit(0);*/


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
        }

        try {
            for (Thread t : threads)
                t.join();
        }
        catch (InterruptedException e) {
            e.printStackTrace();
        }
        ArrayList<ArrayList<Integer>> pixelsImages = new ArrayList<>();
        ArrayList<Integer> trueFalse = new ArrayList<>();
        for (PixelsImagesTrueFalse p : pixelsImagesTrueFalse) {
            pixelsImages.add(p.pixelsImages);
            trueFalse.add(p.trueFalse);
        }
        NeuralNetwork.training(pixelsImages, trueFalse);
    }

    public static BufferedImage leftRightImage(BufferedImage image) {
        int w = image.getWidth();
        int h = image.getHeight();

        int xMin = Integer.MAX_VALUE;
        int yMin = Integer.MAX_VALUE;
        int xMax = Integer.MIN_VALUE;
        int yMax = Integer.MIN_VALUE;
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                if (image.getRGB(j, i) != -1) {
                    if (j < xMin)
                        xMin = j;
                    if (i < yMin)
                        yMin = i;
                    if (j > xMax)
                        xMax = j;
                    if (i > yMax)
                        yMax = i;
                }
            }
        }

        int xShift;
        int yShift;
        if (xMax - xMin >= yMax - yMin) {
            xShift = xMin;
            yShift = yMin - xMin - (int) Math.floor(((xMax - xMin) - (yMax - yMin))/2.);
        }
        else {
            xShift = xMin - yMin - (int) Math.floor(((yMax - yMin) - (xMax - xMin))/2.);
            yShift = yMin;
        }


        if (yShift < 0) {
            for (int i = yMax; i >= yMin; i--) {
                for (int j = xMin; j <= xMax; j++) {
                    if (image.getRGB(j, i) != -1) {
                        image.setRGB(j - xShift, i - yShift, image.getRGB(j, i));
                        image.setRGB(j, i, -1);
                    }
                }
            }
        }
        else if (xShift < 0) {
            for (int i = yMin; i <= yMax; i++) {
                for (int j = xMax; j >= xMin; j--) {
                    if (image.getRGB(j, i) != -1) {
                        image.setRGB(j - xShift, i - yShift, image.getRGB(j, i));
                        image.setRGB(j, i, -1);
                    }
                }
            }
        }
        else {
            for (int i = yMin; i <= yMax; i++) {
                for (int j = xMin; j <= xMax; j++) {
                    if (image.getRGB(j, i) != -1) {
                        image.setRGB(j - xShift, i - yShift, image.getRGB(j, i));
                        image.setRGB(j, i, -1);
                    }
                }
            }
        }

        return image;
    }

    public static BufferedImage scaleImage(BufferedImage before) {
        int w = before.getWidth();
        int h = before.getHeight();

        int xyMax = Integer.MIN_VALUE;
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                if (before.getRGB(j, i) != -1) {
                    if (j > xyMax)
                        xyMax = j;
                    if (i > xyMax)
                        xyMax = i;
                }
            }
        }
        double scale = (double) w / (xyMax + 1);

        BufferedImage after = new BufferedImage(w, h, BufferedImage.TYPE_INT_ARGB);
        AffineTransform at = new AffineTransform();
        at.scale(scale, scale);
        AffineTransformOp scaleOp =
                new AffineTransformOp(at, AffineTransformOp.TYPE_BILINEAR);
        return scaleOp.filter(before, after);
    }

    public static void getObjectOnImage(BufferedImage image, int x, int y, int pixelColor, ArrayList<int[]> imArray) {
        imArray.add(new int[] {x, y});

        if (x != 0 && x != image.getWidth() - 1 && y != 0 && y != image.getHeight() - 1) {
            for (int xi = x - 1; xi <= x + 1; xi++) {
                outer: for (int yi = y - 1; yi <= y + 1; yi++) {
                    int RGB = image.getRGB(xi, yi);
                    if ( !(xi == x && yi == y) && (RGB > pixelColor - 3500000 && RGB < pixelColor + 3500000) ) {
                        for (int[] iA : imArray) {
                            if (iA[0] == xi && iA[1] == yi)
                                continue outer;
                        }
                        getObjectOnImage(image, xi, yi, pixelColor, imArray);
                    }
                }
            }
        }
    }

    public static void listFilesForFolder(final File folder) {
        for (final File fileEntry : Objects.requireNonNull(folder.listFiles())) {
            if (fileEntry.isDirectory()) {
                listFilesForFolder(fileEntry);
            } else {
                images.add(fileEntry);
            }
        }
    }

    public static BufferedImage resize(BufferedImage img, int newW, int newH) {
        Image tmp = img.getScaledInstance(newW, newH, Image.SCALE_SMOOTH);
        BufferedImage dimg = new BufferedImage(newW, newH, BufferedImage.TYPE_INT_ARGB);

        Graphics2D g2d = dimg.createGraphics();
        g2d.drawImage(tmp, 0, 0, null);
        g2d.dispose();

        return dimg;
    }

/*    public static int getMinValueRGB(BufferedImage bufferedImage, int height, int width) {
        int minValue = bufferedImage.getRGB(0, 0);
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                int value = bufferedImage.getRGB(i, j);
                if ( value < minValue)
                    minValue = value;
            }
        }
        return minValue;
    }

    public static int getMaxValueRGB(BufferedImage bufferedImage, int height, int width) {
        int maxValue = bufferedImage.getRGB(0, 0);
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                int value = bufferedImage.getRGB(i, j);
                if ( value > maxValue)
                    maxValue = value;
            }
        }
        return maxValue;
    }

    *//*public static int getClusterMeanValue(BufferedImage bufferedImage, int height, int width) {

        Map<Integer, ArrayList<Cluster>> map = new HashMap<>();
        ArrayList<Integer> clusterMeans = new ArrayList<>();
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                int value = bufferedImage.getRGB(i, j);
                for (Integer clusterMean : clusterMeans) {
                    if (clusterMean - 1000 < value && value < clusterMean + 1000)
                }
                Cluster cluster = new Cluster(i, j, value);


            }
        }
    }*//*

    static class Cluster {
        private int x;
        private int y;
        private int value;

        public Cluster(int x, int y, int value) {
            this.x = x;
            this.y = y;
            this.value = value;
        }

        public int getX() {
            return x;
        }

        public int getY() {
            return y;
        }

        public int getMean() {
            return value;
        }
    }*/

/*    static class NeuralNetwork {

        private static int[][] trainingInputs = new int[][] {new int[] {0, 0, 1},
                new int[] {1, 1, 1},
                new int[] {1, 0, 1},
                new int[] {0, 1, 1}};
        private static int[] trainingOutputs = new int[] {0, 1, 1, 0};

        private static int numberOfSynaptic = 3;
        private static ArrayList<Double> synapticWeights = new ArrayList<>();

        public NeuralNetwork() {
            for (int i = 0; i < NeuralNetwork.numberOfSynaptic; i++)
                synapticWeights.add(new Random().nextDouble());
//            System.out.println(synapticWeights);

            int n = 1000000;
            double[] output = new double[trainingOutputs.length];
            System.out.println(Arrays.toString(output));
            for (int i = 0; i < n; i++) {
                output = new double[trainingOutputs.length];
                for (int j = 0; j < trainingInputs.length; j++) {
                    for (int l = 0; l < numberOfSynaptic; l++)
                        output[j] += trainingInputs[j][l]*synapticWeights.get(l);

                    if (Math.round(output[j]) > trainingOutputs[j]) {
                        for (int s = 0; s < NeuralNetwork.numberOfSynaptic; s++)
                            synapticWeights.set( s, synapticWeights.get(s) - (output[j] - trainingOutputs[j]) / (2 * new Random().nextDouble() + 2));
                    }
                    else if (Math.round(output[j]) < trainingOutputs[j]) {
                        for (int s = 0; s < NeuralNetwork.numberOfSynaptic; s++)
                            synapticWeights.set( s, synapticWeights.get(s) + (trainingOutputs[j] - output[j]) / (2 * new Random().nextDouble() + 2));
                    }
                }

                if (Math.round(output[0]) == trainingOutputs[0] &&
                        Math.round(output[1]) == trainingOutputs[1] &&
                        Math.round(output[2]) == trainingOutputs[2] &&
                        Math.round(output[3]) == trainingOutputs[3]) {
                    System.out.println("\n\ni = " + i + "\n");
                    break;
                }
            }
            System.out.println(Arrays.toString(output));
            for (double el : output)
                System.out.println(Math.round(el));
        }
    }*/

    /*static class NeuralNetwork {

        private static final ArrayList<Double> synapticWeights = new ArrayList<>();

        public NeuralNetwork(ArrayList<ArrayList<Integer>> trainingInputs,
                             ArrayList<Integer> trainingOutputs,
                             int numberOfSynaptic) {

            for (int i = 0; i < numberOfSynaptic; i++)
                synapticWeights.add(2 * new Random().nextDouble() - 1);

            int n = 100;
            double[] output = new double[trainingOutputs.size()];
            for (int i = 0; i < n; i++) {
                output = new double[trainingOutputs.size()];
                boolean checkEquals = true;
                for (int j = 0; j < trainingInputs.size(); j++) {
                    for (int l = 0; l < numberOfSynaptic; l++)
                        output[j] += (double) trainingInputs.get(j).get(l) * synapticWeights.get(l);

                    if (Math.round(output[j]) != trainingOutputs.get(j)) {
                        for (int s = 0; s < numberOfSynaptic; s++) {
                            double numerator = output[j] - trainingOutputs.get(j);
                            double denominator = 2 * new Random().nextDouble() + (numberOfSynaptic - 1);
                            float fraction = (float) (numerator*1.f / denominator);
                            float sW = (float) (synapticWeights.get(s) + 0.f);
                            double newWeight = sW + fraction;
                            synapticWeights.set(s, newWeight);
//                            synapticWeights.set(s, synapticWeights.get(s) + (output[j] - trainingOutputs.get(j)) / (2 * new Random().nextDouble() + (numberOfSynaptic - 1)));
                        }
                        checkEquals = false;
                    }
                    if (j == 4)
                        System.exit(0);
                }

                if (checkEquals) {
                    System.out.println("!!! checkEquals !!!");
                    break;
                }
            }

            int stat = 0;
            for (int i = 0; i < output.length; i++) {
                if (Math.round(output[i]) == trainingOutputs.get(i))
                    stat++;
            }

            System.out.println("Stat: " + stat/output.length);
            System.out.println("========== END ==========");
        }
    }*/

    static class NeuralNetwork {

        private static final ArrayList<BigDecimal> synapticWeights = new ArrayList<>();

        public static void training(ArrayList<ArrayList<Integer>> trainingInputs,
                             ArrayList<Integer> trainingOutputs) {

            int numberOfSynaptic = trainingInputs.get(0).size();
            MathContext mathContext20 = new MathContext(20); // для BigDecimal

            for (int i = 0; i < numberOfSynaptic; i++)
                synapticWeights.add(BigDecimal.valueOf(2 * Math.random() - 1));
            /*for (int i = 0; i < numberOfSynaptic; i++)
                synapticWeights.add(BigDecimal.valueOf(0.0002*i - 1));*/

            System.out.println("synapticWeights");

            int n = 20;
            int size = trainingOutputs.size();
            BigDecimal[] output = new BigDecimal[size];
            /* 3 вариант
             **/
            BigDecimal epsilon = BigDecimal.valueOf(0.7); // скорость обучения (0.7)
            BigDecimal alpha = BigDecimal.valueOf(0.3); // момент (0.3)
            BigDecimal[] deltaWeights = new BigDecimal[numberOfSynaptic];
            Arrays.fill(deltaWeights, BigDecimal.ZERO);
            for (int i = 0; i < n; i++) {
                output = new BigDecimal[size];
                Arrays.fill(output, BigDecimal.valueOf(0));
                BigInteger statistic = BigInteger.valueOf(size);

                BigDecimal error = BigDecimal.ZERO;

                for (int j = 0; j < size; j++) {
                    for (int l = 0; l < numberOfSynaptic; l++)
                        output[j] = output[j].add(synapticWeights.get(l).multiply(BigDecimal.valueOf(trainingInputs.get(j).get(l)), mathContext20), mathContext20);

                    double outputDouble = sigmoid(output[j].doubleValue());
                    if ((int) Math.round(outputDouble) != trainingOutputs.get(j)) {

                        error = error.add(BigDecimal.valueOf(Math.pow(trainingOutputs.get(j) - outputDouble, 2)/numberOfSynaptic), mathContext20);

                        BigDecimal delta = BigDecimal.valueOf(Math.pow(trainingOutputs.get(j) - outputDouble, 2)/numberOfSynaptic);
                        /* 3 вариант
                         *
                        BigDecimal delta = BigDecimal.valueOf(((double) trainingOutputs.get(j) - outputDouble)*outputDouble*(1. - outputDouble));*/
                        /* 2 вариант
                         *
                        BigDecimal er = BigDecimal.valueOf((trainingOutputs.get(j) - outputDouble)*outputDouble*(1. - outputDouble));*/
                        /* 1 вариант
                         *
                        BigDecimal err = BigDecimal.valueOf(Math.pow(trainingOutputs.get(j) - outputDouble, 2)/numberOfSynaptic);*/

                        for (int s = 0; s < numberOfSynaptic; s++) {

                            /* 3 вариант
                             **/
                            BigDecimal grad = BigDecimal.valueOf(-trainingInputs.get(j).get(s)).multiply(delta, mathContext20);
                            deltaWeights[s] = epsilon.multiply(grad, mathContext20).add(alpha.multiply(deltaWeights[s], mathContext20), mathContext20);
                            if ((int) Math.round(outputDouble) < trainingOutputs.get(j))
                                synapticWeights.set(s, synapticWeights.get(s).subtract(deltaWeights[s], mathContext20));
                            else
                                synapticWeights.set(s, synapticWeights.get(s).add(deltaWeights[s], mathContext20));
                            /* 2 вариант
                             *
                            BigDecimal err = BigDecimal.valueOf(trainingInputs.get(j).get(s)).multiply(er, mathContext20);
                            synapticWeights.set(s, synapticWeights.get(s).subtract(err, mathContext20));*/
                            /* 1 вариант
                             *
                            if ((int) Math.round(outputDouble) < trainingOutputs.get(j))
                                synapticWeights.set(s, synapticWeights.get(s).subtract(err, mathContext20));
                            else
                                synapticWeights.set(s, synapticWeights.get(s).add(err, mathContext20));*/
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

//            System.out.println("trainingOutput: " + trainingOutputs);

            System.out.print("FINAL_OUTPUT: ");
            Arrays.fill(output, BigDecimal.ZERO);
            long stat = 0;
            for (int i = 0; i < size; i++) {
                for (int j = 0; j < numberOfSynaptic; j++) {
                    output[i] = output[i].add(BigDecimal.valueOf(trainingInputs.get(i).get(j)).multiply(synapticWeights.get(j), mathContext20), mathContext20);
                }
                int intOutput = (int) Math.round(sigmoid(output[i].doubleValue()));
                if (intOutput == trainingOutputs.get(i))
                    stat++;
//                System.out.print(intOutput + " ");
            }
            System.out.println();
            System.out.println("Stat: " + (double) stat/size);

            System.out.println("========== END ==========");

            long stat2 = 0;
            ArrayList<Integer> testingOutputs = testing(trainingInputs, synapticWeights);
            for (int i = 0; i < size; i++) {
                if (testingOutputs.get(i).equals(trainingOutputs.get(i)))
                    stat2++;
            }
            System.out.println("Stat2: " + (double) stat2/size);
        }

        public static double sigmoid(double x) {
//            return 1. / (1. + Math.pow(Math.E, -x));

            if (x <= -6.91)
                return 0.;
            else if (x <= -2.2)
                return 0.1;
            else if (x <= -1.39)
                return 0.2;
            else if (x <= -0.85)
                return 0.3;
            else if (x <= -0.41)
                return 0.4;
            else if (x >= 0.41)
                return 0.6;
            else if (x >= 0.85)
                return 0.7;
            else if (x >= 1.39)
                return 0.8;
            else if (x >= 2.2)
                return 0.9;
            else if (x >= 6.91)
                return 1.;
            else
                return 0.5;
        }

        public static ArrayList<Integer> testing(ArrayList<ArrayList<Integer>> inputs, ArrayList<BigDecimal> weights) {
            int size = inputs.size();
            int size2 = inputs.get(0).size();
            BigDecimal[] outputs = new BigDecimal[size];
            Arrays.fill(outputs, BigDecimal.ZERO);
            MathContext mathContext20 = new MathContext(20);
            ArrayList<Integer> result = new ArrayList<>();
            for (int i = 0; i < size; i++) {
                for (int j = 0; j < size2; j++)
                    outputs[i] = outputs[i].add(BigDecimal.valueOf(inputs.get(i).get(j)).multiply(weights.get(j), mathContext20), mathContext20);
                result.add( (int) Math.round(sigmoid(outputs[i].doubleValue())) );
            }
            return result;
        }
    }

    static class PixelsImagesTrueFalse {
        private final ArrayList<Integer> pixelsImages;
        private final int trueFalse;

        public PixelsImagesTrueFalse(ArrayList<Integer> pixelsImages, int trueFalse) {
            this.pixelsImages = pixelsImages;
            this.trueFalse = trueFalse;
        }
    }
}
