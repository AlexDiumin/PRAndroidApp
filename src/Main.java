import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.math.BigDecimal;
import java.math.MathContext;
import java.util.ArrayList;
import java.util.Objects;

public class Main {

    private static final MathContext mathContext20 = new MathContext(20);

    public static void main(String[] args) {

        TensorSize inputSize = new TensorSize(3, 96, 96); // размер входного тензора
        Tensor input = new Tensor(inputSize);                               // входной тензор
        ArrayList<Tensor> inputTensors = new ArrayList<>();                 // массив входных тензоров
        TensorSize outputSize = new TensorSize(1, 1, 1);  // размер выходного тензора
        Tensor output = new Tensor(outputSize);                             // выходной тензор
        ArrayList<Tensor> outputTensors = new ArrayList<>();                // массив выходных тензоров

        ArrayList<BufferedImage> bufferedImages0 = new ArrayList<>();
        listFilesForFolder(new File("0"), bufferedImages0);
        int bufferedImagesSize0 = bufferedImages0.size();
        for (int i = bufferedImagesSize0 - 1; i >= 0; i--) {
            input = new Tensor(inputSize);
            output = new Tensor(outputSize);
            for (int h = 0; h < inputSize.height; h++) {
                for (int w = 0; w < inputSize.width; w++) {
                    input.setByIndex(0, h, w, BigDecimal.valueOf(new Color(bufferedImages0.get(i).getRGB(h, w)).getRed()));
                    input.setByIndex(1, h, w, BigDecimal.valueOf(new Color(bufferedImages0.get(i).getRGB(h, w)).getGreen()));
                    input.setByIndex(2, h, w, BigDecimal.valueOf(new Color(bufferedImages0.get(i).getRGB(h, w)).getBlue()));
                }
            }
            inputTensors.add(input);
            output.setByIndex(0, BigDecimal.ZERO);
            outputTensors.add(output);

            bufferedImages0.get(i).flush();
            bufferedImages0.remove(i);
        }
        bufferedImages0.clear();

        ArrayList<BufferedImage> bufferedImages1 = new ArrayList<>();
        listFilesForFolder(new File("1"), bufferedImages1);
        int bufferedImagesSize1 = bufferedImages1.size();
        for (int i = bufferedImagesSize1 - 1; i >= 0; i--) {
            input = new Tensor(inputSize);
            output = new Tensor(outputSize);
            for (int h = 0; h < inputSize.height; h++) {
                for (int w = 0; w < inputSize.width; w++) {
                    input.setByIndex(0, h, w, BigDecimal.valueOf(new Color(bufferedImages1.get(i).getRGB(h, w)).getRed()));
                    input.setByIndex(1, h, w, BigDecimal.valueOf(new Color(bufferedImages1.get(i).getRGB(h, w)).getGreen()));
                    input.setByIndex(2, h, w, BigDecimal.valueOf(new Color(bufferedImages1.get(i).getRGB(h, w)).getBlue()));
                }
            }
            inputTensors.add(input);
            output.setByIndex(0, BigDecimal.ONE);
            outputTensors.add(output);

            bufferedImages1.get(i).flush();
            bufferedImages1.remove(i);
        }
        bufferedImages1.clear();


        int N = 20; // кол-во эпох
        int imagesCount = inputTensors.size();
        ConvolutionalLayer convolutionalLayer1 = new ConvolutionalLayer(inputSize, 3, 5, 0, 1);
        SigmoidLayer sigmoidLayer1 = new SigmoidLayer(new TensorSize(3, 92, 92));
        MaxPoolingLayer maxPoolingLayer1 = new MaxPoolingLayer(new TensorSize(3, 92, 92));
        ConvolutionalLayer convolutionalLayer2 = new ConvolutionalLayer(new TensorSize(3, 46, 46), 3, 5, 0, 1);
        SigmoidLayer sigmoidLayer2 = new SigmoidLayer(new TensorSize(3, 42, 42));
        MaxPoolingLayer maxPoolingLayer2 = new MaxPoolingLayer(new TensorSize(3, 42, 42));
        FullyConnectedLayer fullyConnectedLayer1 = new FullyConnectedLayer(new TensorSize(3, 21, 21), 1, "sigmoid");
        // прямое распространение
        Tensor convolutionalForwardOutput1;
        Tensor sigmoidForwardOutput1;
        Tensor maxPoolingForwardOutput1;
        Tensor convolutionalForwardOutput2;
        Tensor sigmoidForwardOutput2;
        Tensor maxPoolingForwardOutput2;
        Tensor fullyConnectedForwardOutput1;
        // обратное распространение
        Tensor convolutionalBackwardOutput1;
        Tensor sigmoidBackwardOutput1;
        Tensor maxPoolingBackwardOutput1;
        Tensor convolutionalBackwardOutput2;
        Tensor sigmoidBackwardOutput2;
        Tensor maxPoolingBackwardOutput2;
        Tensor fullyConnectedBackwardOutput1;
        BigDecimal error; // ошибка для статистики
        int statistic; // статистика
        BigDecimal learningRate = BigDecimal.valueOf(0.5); // скорость обучения
        for (int n = 0; n < N; n++) {
            error = BigDecimal.ZERO;
            statistic = imagesCount;
            for (int i = 0; i < imagesCount; i++) {

                // прямое распространение
                convolutionalForwardOutput1 = convolutionalLayer1.forward(inputTensors.get(i));        // 1 сверточный слой
                sigmoidForwardOutput1 = sigmoidLayer1.forward(convolutionalForwardOutput1);            // 1 активационный слой
                maxPoolingForwardOutput1 = maxPoolingLayer1.forward(sigmoidForwardOutput1);            // 1 подвыборочный слой
                convolutionalForwardOutput2 = convolutionalLayer2.forward(maxPoolingForwardOutput1);   // 2 сверточный слой
                sigmoidForwardOutput2 = sigmoidLayer2.forward(convolutionalForwardOutput2);            // 2 активационный слой
                maxPoolingForwardOutput2 = maxPoolingLayer2.forward(sigmoidForwardOutput2);            // 2 подвыборочный слой
                fullyConnectedForwardOutput1 = fullyConnectedLayer1.forward(maxPoolingForwardOutput2); // 1 полносвязный слой

                // расчет ошибки для статистики
                if (fullyConnectedForwardOutput1.getByIndex(0).compareTo(outputTensors.get(i).getByIndex(0)) != 0) {
                    error = error.add(((outputTensors.get(i).getByIndex(0).subtract(fullyConnectedForwardOutput1.getByIndex(0), mathContext20)).pow(2)).divide(BigDecimal.valueOf(imagesCount), mathContext20), mathContext20);
                    statistic--;
                }

                // обратное распространение
                fullyConnectedBackwardOutput1 = fullyConnectedLayer1.backward(fullyConnectedForwardOutput1, maxPoolingForwardOutput2);
                maxPoolingBackwardOutput2 = maxPoolingLayer2.backward(fullyConnectedBackwardOutput1, sigmoidForwardOutput2);
                sigmoidBackwardOutput2 = sigmoidLayer2.backward(maxPoolingBackwardOutput2, convolutionalForwardOutput2);
                convolutionalBackwardOutput2 = convolutionalLayer2.backward(sigmoidBackwardOutput2, maxPoolingForwardOutput1);
                maxPoolingBackwardOutput1 = maxPoolingLayer1.backward(convolutionalBackwardOutput2, sigmoidForwardOutput1);
                sigmoidBackwardOutput1 = sigmoidLayer1.backward(maxPoolingBackwardOutput1, convolutionalForwardOutput1);
                convolutionalBackwardOutput1 = convolutionalLayer1.backward(sigmoidBackwardOutput1, inputTensors.get(i));

                // обновление весовых коэффициентов
                convolutionalLayer1.updateWeights(learningRate);
                convolutionalLayer2.updateWeights(learningRate);
                fullyConnectedLayer1.updateWeights(learningRate);
            }

            System.out.println(n + " Error: " + error);
            System.out.println(n + " Statistic: " + (double) statistic/imagesCount);
            System.out.println();
        }
    }

    public static void listFilesForFolder(final File folder, ArrayList<BufferedImage> bufferedImages) {
        for (final File fileEntry : Objects.requireNonNull(folder.listFiles())) {
            if (fileEntry.isDirectory())
                listFilesForFolder(fileEntry, bufferedImages);
            else {
                try {
                    BufferedImage bufferedImage = ImageIO.read(fileEntry);
                    if ( !(bufferedImage.getWidth() == 96 && bufferedImage.getHeight() == 96) )
                        bufferedImage = resize(bufferedImage, 96, 96);

                    bufferedImages.add(bufferedImage);

                    bufferedImage.flush();
                }
                catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }

    public static BufferedImage resize(BufferedImage img, int newW, int newH) {
        Image tmp = img.getScaledInstance(newW, newH, Image.SCALE_SMOOTH);
        BufferedImage resizedImage = new BufferedImage(newW, newH, BufferedImage.TYPE_INT_ARGB);

        Graphics2D g2d = resizedImage.createGraphics();
        g2d.drawImage(tmp, 0, 0, null);
        g2d.dispose();

        return resizedImage;
    }
}
