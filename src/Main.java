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
                    input.setByIndex(0, h, w, BigDecimal.valueOf(new Color(bufferedImages0.get(i).getRGB(h, w)).getRed()/255.));
                    input.setByIndex(1, h, w, BigDecimal.valueOf(new Color(bufferedImages0.get(i).getRGB(h, w)).getGreen()/255.));
                    input.setByIndex(2, h, w, BigDecimal.valueOf(new Color(bufferedImages0.get(i).getRGB(h, w)).getBlue()/255.));
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
                    input.setByIndex(0, h, w, BigDecimal.valueOf(new Color(bufferedImages1.get(i).getRGB(h, w)).getRed()/255.));
                    input.setByIndex(1, h, w, BigDecimal.valueOf(new Color(bufferedImages1.get(i).getRGB(h, w)).getGreen()/255.));
                    input.setByIndex(2, h, w, BigDecimal.valueOf(new Color(bufferedImages1.get(i).getRGB(h, w)).getBlue()/255.));
                }
            }
            inputTensors.add(input);
            output.setByIndex(0, BigDecimal.ONE);
            outputTensors.add(output);

            bufferedImages1.get(i).flush();
            bufferedImages1.remove(i);
        }
        bufferedImages1.clear();


        int N = 2000; // кол-во эпох
        int imagesCount = inputTensors.size(); // кол-во входящих изображений/тензоров
        int channelsCount = 1; // кол-во каналов/признаков
        ConvolutionalLayer[] convolutionalLayer1 = new ConvolutionalLayer[channelsCount]; // = new ConvolutionalLayer(inputSize, 3, 5, 0, 1);
        SigmoidLayer[] sigmoidLayer1 = new SigmoidLayer[channelsCount];// = new SigmoidLayer(new TensorSize(3, 92, 92));
        MaxPoolingLayer[] maxPoolingLayer1 = new MaxPoolingLayer[channelsCount]; // = new MaxPoolingLayer(new TensorSize(3, 92, 92));
        ConvolutionalLayer[] convolutionalLayer2 = new ConvolutionalLayer[channelsCount]; // new ConvolutionalLayer(new TensorSize(3, 46, 46), 3, 5, 0, 1);
        SigmoidLayer[] sigmoidLayer2 = new SigmoidLayer[channelsCount]; // = new SigmoidLayer(new TensorSize(3, 42, 42));
        MaxPoolingLayer[] maxPoolingLayer2 = new MaxPoolingLayer[channelsCount]; // = new MaxPoolingLayer(new TensorSize(3, 42, 42));
        FullyConnectedLayer[] fullyConnectedLayer1 = new FullyConnectedLayer[channelsCount]; // = new FullyConnectedLayer(new TensorSize(3, 21, 21), 1, "sigmoid");
        for (int i = 0; i < channelsCount; i++) {
            convolutionalLayer1[i] = new ConvolutionalLayer(inputSize, 3, 5, 0, 1);
            sigmoidLayer1[i] = new SigmoidLayer(new TensorSize(3, 92, 92));
            maxPoolingLayer1[i] = new MaxPoolingLayer(new TensorSize(3, 92, 92));
            convolutionalLayer2[i] = new ConvolutionalLayer(new TensorSize(3, 46, 46), 3, 5, 0, 1);
            sigmoidLayer2[i] = new SigmoidLayer(new TensorSize(3, 42, 42));
            maxPoolingLayer2[i] = new MaxPoolingLayer(new TensorSize(3, 42, 42));
            fullyConnectedLayer1[i] = new FullyConnectedLayer(new TensorSize(3, 21, 21), 1, "sigmoid");
        }
        FullyConnectedLayer fullyConnectedLayer2 = new FullyConnectedLayer(new TensorSize(1, 1, channelsCount), 1, "sigmoid");
        // прямое распространение
        Tensor[] convolutionalForwardOutput1 = new Tensor[channelsCount];
        Tensor[] sigmoidForwardOutput1 = new Tensor[channelsCount];
        Tensor[] maxPoolingForwardOutput1 = new Tensor[channelsCount];
        Tensor[] convolutionalForwardOutput2 = new Tensor[channelsCount];
        Tensor[] sigmoidForwardOutput2 = new Tensor[channelsCount];
        Tensor[] maxPoolingForwardOutput2 = new Tensor[channelsCount];
        Tensor[] fullyConnectedForwardOutput1 = new Tensor[channelsCount];
        Tensor fullyConnectedForwardOutput2;
        Tensor dOut = new Tensor(1, 1, 1);
        // обратное распространение
        Tensor[] convolutionalBackwardOutput1 = new Tensor[channelsCount];
        Tensor[] sigmoidBackwardOutput1 = new Tensor[channelsCount];
        Tensor[] maxPoolingBackwardOutput1 = new Tensor[channelsCount];
        Tensor[] convolutionalBackwardOutput2 = new Tensor[channelsCount];
        Tensor[] sigmoidBackwardOutput2 = new Tensor[channelsCount];
        Tensor[] maxPoolingBackwardOutput2 = new Tensor[channelsCount];
        Tensor[] fullyConnectedBackwardOutput1 = new Tensor[channelsCount];
        Tensor fullyConnectedBackwardOutput2;
        BigDecimal error; // ошибка для статистики
        int statistic; // статистика
        BigDecimal learningRate = BigDecimal.valueOf(0.7); // скорость обучения
        for (int n = 0; n < N; n++) {
            dOut.setByIndex(0, BigDecimal.ZERO);
            error = BigDecimal.ZERO;
            statistic = imagesCount;
            for (int i = 0; i < imagesCount; i++) {

                // прямое распространение
                for (int j = 0; j < channelsCount; j++) {
                    convolutionalForwardOutput1[j] = convolutionalLayer1[j].forward(inputTensors.get(i));           // 1 сверточный слой
                    sigmoidForwardOutput1[j] = sigmoidLayer1[j].forward(convolutionalForwardOutput1[j]);            // 1 активационный слой
                    maxPoolingForwardOutput1[j] = maxPoolingLayer1[j].forward(sigmoidForwardOutput1[j]);            // 1 подвыборочный слой
                    convolutionalForwardOutput2[j] = convolutionalLayer2[j].forward(maxPoolingForwardOutput1[j]);   // 2 сверточный слой
                    sigmoidForwardOutput2[j] = sigmoidLayer2[j].forward(convolutionalForwardOutput2[j]);            // 2 активационный слой
                    maxPoolingForwardOutput2[j] = maxPoolingLayer2[j].forward(sigmoidForwardOutput2[j]);            // 2 подвыборочный слой
                    fullyConnectedForwardOutput1[j] = fullyConnectedLayer1[j].forward(maxPoolingForwardOutput2[j]); // 1 полносвязный слой
                }
                fullyConnectedForwardOutput2 = fullyConnectedLayer2.forward(Tensor.arrayToTensor(fullyConnectedForwardOutput1));

                // расчет ошибки для статистики
                if (fullyConnectedForwardOutput2.getByIndex(0).compareTo(outputTensors.get(i).getByIndex(0)) != 0) {
                    error = error.add(((outputTensors.get(i).getByIndex(0).subtract(fullyConnectedForwardOutput2.getByIndex(0), mathContext20)).pow(2)).divide(BigDecimal.valueOf(imagesCount), mathContext20), mathContext20);
                    statistic--;;
                }

                // градиент на выходе
                dOut.setByIndex(0, (fullyConnectedForwardOutput2.getByIndex(0).subtract(outputTensors.get(i).getByIndex(0), mathContext20)).multiply(fullyConnectedForwardOutput2.getByIndex(0), mathContext20).multiply(BigDecimal.ONE.subtract(fullyConnectedForwardOutput2.getByIndex(0), mathContext20)) );

                // обратное распространение
                for (int j = 0; j < channelsCount; j++) {
                    fullyConnectedBackwardOutput2 = fullyConnectedLayer2.backward(dOut, Tensor.arrayToTensor(fullyConnectedForwardOutput1));
                    fullyConnectedBackwardOutput1[j] = fullyConnectedLayer1[j].backward(fullyConnectedBackwardOutput2, maxPoolingForwardOutput2[j]);
                    maxPoolingBackwardOutput2[j] = maxPoolingLayer2[j].backward(fullyConnectedBackwardOutput1[j], sigmoidForwardOutput2[j]);
                    sigmoidBackwardOutput2[j] = sigmoidLayer2[j].backward(maxPoolingBackwardOutput2[j], convolutionalForwardOutput2[j]);
                    convolutionalBackwardOutput2[j] = convolutionalLayer2[j].backward(sigmoidBackwardOutput2[j], maxPoolingForwardOutput1[j]);
                    maxPoolingBackwardOutput1[j] = maxPoolingLayer1[j].backward(convolutionalBackwardOutput2[j], sigmoidForwardOutput1[j]);
                    sigmoidBackwardOutput1[j] = sigmoidLayer1[j].backward(maxPoolingBackwardOutput1[j], convolutionalForwardOutput1[j]);
                    convolutionalBackwardOutput1[j] = convolutionalLayer1[j].backward(sigmoidBackwardOutput1[j], inputTensors.get(i));
                }

                // обновление весовых коэффициентов
                for (int j = 0; j < channelsCount; j++) {
                    convolutionalLayer1[j].updateWeights(learningRate);
                    convolutionalLayer2[j].updateWeights(learningRate);
                    fullyConnectedLayer1[j].updateWeights(learningRate);
                }
                fullyConnectedLayer2.updateWeights(learningRate);
            }

            System.out.println(n + " Error: " + error);
            System.out.println(n + " Statistic: " + (double) statistic/imagesCount);
            System.out.println();

            /*if (error.equals(BigDecimal.ZERO) || statistic == imagesCount) {
                System.out.println("!!! CONGRATULATION !!!");
                break;
            }*/
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
