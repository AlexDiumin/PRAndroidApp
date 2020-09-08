import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.math.BigDecimal;
import java.math.MathContext;
import java.util.ArrayList;

public class NeuralNetwork {

    private static final MathContext mathContext20 = new MathContext(20);

    private final TensorSize inputSize; // размер входного тензора
    private final TensorSize outputSize; // размер выходного тензора

    private final Convolution[] convolutions; // сверточные слои
    private final MaxPooling[] maxPoolings; // слои подвыборки/субдискретизации
    private final FullyNected[] fullyNecteds; // полносвязные слои

    private final Tensor[] forwardOutputs; // выходные тензоры прямого распространения

    // гиперпараметры
    private final BigDecimal epsilon; // скорость обучения
    private final BigDecimal alpha; // момент

    /*public NeuralNetwork(final TensorSize inputSize, int classesCount) {
        this.outputSize = new TensorSize(1, 1, classesCount); // запоминаем размер выходного тензора

        int filterSize = 3; // размер фильтра для сверточных слоев
        int stride = 1; // шаг фильтра для сверточных слоев
        int padding = 1; // размер отступов для сверточных слоев
        int scale = 2; // коэффициент уменьшения для подвыборочных слоев

        this.convolutions = new Convolution[13];
        this.maxPoolings = new MaxPooling[5];
        this.fullyNecteds = new FullyNected[3];

        this.convolutions[0] = new Convolution(inputSize, filterSize, 64, stride, padding);
        this.convolutions[1] = new Convolution(this.convolutions[0].getOutputSize(), filterSize, 64, stride, padding);

        this.maxPoolings[0] = new MaxPooling(this.convolutions[1].getOutputSize(), scale);

        this.convolutions[2] = new Convolution(this.maxPoolings[0].getOutputSize(), filterSize, 128, stride, padding);
        this.convolutions[3] = new Convolution(this.convolutions[2].getOutputSize(), filterSize, 128, stride, padding);

        this.maxPoolings[1] = new MaxPooling(this.convolutions[3].getOutputSize(), scale);

        this.convolutions[4] = new Convolution(this.maxPoolings[1].getOutputSize(), filterSize, 256, stride, padding);
        this.convolutions[5] = new Convolution(this.convolutions[4].getOutputSize(), filterSize, 256, stride, padding);
        this.convolutions[6] = new Convolution(this.convolutions[5].getOutputSize(), filterSize, 256, stride, padding);

        this.maxPoolings[2] = new MaxPooling(this.convolutions[6].getOutputSize(), scale);

        this.convolutions[7] = new Convolution(this.maxPoolings[2].getOutputSize(), filterSize, 512, stride, padding);
        this.convolutions[8] = new Convolution(this.convolutions[7].getOutputSize(), filterSize, 512, stride, padding);
        this.convolutions[9] = new Convolution(this.convolutions[8].getOutputSize(), filterSize, 512, stride, padding);

        this.maxPoolings[3] = new MaxPooling(this.convolutions[9].getOutputSize(), scale);

        this.convolutions[10] = new Convolution(this.maxPoolings[3].getOutputSize(), filterSize, 512, stride, padding);
        this.convolutions[11] = new Convolution(this.convolutions[10].getOutputSize(), filterSize, 512, stride, padding);
        this.convolutions[12] = new Convolution(this.convolutions[11].getOutputSize(), filterSize, 512, stride, padding);

        this.maxPoolings[4] = new MaxPooling(this.convolutions[12].getOutputSize(), scale);

        this.fullyNecteds[0] = new FullyNected(this.maxPoolings[4].getOutputSize(), new TensorSize(1, 1, 4096));
        this.fullyNecteds[1] = new FullyNected(this.fullyNecteds[0].getOutputSize(), new TensorSize(1, 1, 4096));
        this.fullyNecteds[2] = new FullyNected(this.fullyNecteds[1].getOutputSize(), this.outputSize);

        this.forwardOutputs = new Tensor[13 + 5 + 3];

        this.epsilon = BigDecimal.valueOf(0.7);
        this.alpha = BigDecimal.valueOf(0.07);
    }*/
    public NeuralNetwork(final TensorSize inputSize, int classesCount) {
        this.inputSize = inputSize; // запоминаем размер выходного тензора
        this.outputSize = new TensorSize(1, 1, classesCount); // запоминаем размер выходного тензора

        int filterSize = 3; // размер фильтра для сверточных слоев
        int stride = 1; // шаг фильтра для сверточных слоев
        int padding = 1; // размер отступов для сверточных слоев
        int scale = 2; // коэффициент уменьшения для подвыборочных слоев

        this.convolutions = new Convolution[8];
        this.maxPoolings = new MaxPooling[5];
        this.fullyNecteds = new FullyNected[3];

        this.convolutions[0] = new Convolution(this.inputSize, filterSize, 2, stride, padding);

        this.maxPoolings[0] = new MaxPooling(this.convolutions[0].getOutputSize(), scale);

        this.convolutions[1] = new Convolution(this.maxPoolings[0].getOutputSize(), filterSize, 4, stride, padding);

        this.maxPoolings[1] = new MaxPooling(this.convolutions[1].getOutputSize(), scale);

        this.convolutions[2] = new Convolution(this.maxPoolings[1].getOutputSize(), filterSize, 8, stride, padding);
        this.convolutions[3] = new Convolution(this.convolutions[2].getOutputSize(), filterSize, 8, stride, padding);

        this.maxPoolings[2] = new MaxPooling(this.convolutions[3].getOutputSize(), scale);

        this.convolutions[4] = new Convolution(this.maxPoolings[2].getOutputSize(), filterSize, 16, stride, padding);
        this.convolutions[5] = new Convolution(this.convolutions[4].getOutputSize(), filterSize, 16, stride, padding);

        this.maxPoolings[3] = new MaxPooling(this.convolutions[5].getOutputSize(), scale);

        this.convolutions[6] = new Convolution(this.maxPoolings[3].getOutputSize(), filterSize, 16, stride, padding);
        this.convolutions[7] = new Convolution(this.convolutions[6].getOutputSize(), filterSize, 16, stride, padding);

        this.maxPoolings[4] = new MaxPooling(this.convolutions[7].getOutputSize(), scale);

        this.fullyNecteds[0] = new FullyNected(this.maxPoolings[4].getOutputSize(), new TensorSize(1, 1, 8));
        this.fullyNecteds[1] = new FullyNected(this.fullyNecteds[0].getOutputSize(), new TensorSize(1, 1, 8));
        this.fullyNecteds[2] = new FullyNected(this.fullyNecteds[1].getOutputSize(), this.outputSize, true);

        this.forwardOutputs = new Tensor[8 + 5 + 3];

        this.epsilon = BigDecimal.valueOf(0.7);
        this.alpha = BigDecimal.valueOf(0.007);
    }

    public void training(final ArrayList<String> inputPaths, final ArrayList<Integer> classNumbers) {
        Tensor input; // тензор для хранения входных изображений
        Tensor trueOutput = new Tensor(this.outputSize); // тензор для хранения идеальных выходов
        Tensor networkOutput; // тензор выходов нейронной сети
        Tensor deltasOut = new Tensor(this.outputSize); // тензор выходных дельт
        BigDecimal error;
        int statistic;
        int N = 2000; // кол-во эпох
        // проходимся по эпохам
        for (int n = 0; n < N; n++) {
            error = BigDecimal.ZERO;
            statistic = 0;
            // проходимся по входным тензорам
            for (int i = 0; i < inputPaths.size(); i++) {
                input = this.getTensorFromImage(inputPaths.get(i), this.inputSize); // инициализируем входной тензор
                // инициализируем идеальный выходной тензор
                for (int j = 1; j <= this.outputSize.depth*this.outputSize.height*this.outputSize.width; j++) {
                    if (j == classNumbers.get(i))
                        trueOutput.setByIndex(0, 0, j - 1, BigDecimal.ONE);
                    else
                        trueOutput.setByIndex(0, 0, j - 1, BigDecimal.ZERO);
                }

                networkOutput = this.forward(input); // прямое распространение + получаем выход сети

                // проходимся по выходному тензору
                for (int d = 0; d < this.outputSize.depth; d++) {
                    for (int h = 0; h < this.outputSize.height; h++) {
                        for (int w = 0; w < this.outputSize.width; w++) {
                            if (trueOutput.getByIndex(d, h, w).compareTo(BigDecimal.ONE) == 0)
                                networkOutput.setByIndex(d, h, w, BigDecimal.ONE.subtract(networkOutput.getByIndex(d, h, w), mathContext20));

                            // считаем дельты по выходу
                            deltasOut.setByIndex(d, h, w, trueOutput.getByIndex(d, h, w)
                                    .subtract(networkOutput.getByIndex(d, h, w), mathContext20));



                            System.out.println(deltasOut.getByIndex(d, h, w));



                            error = error.add(BigDecimal.valueOf(Math.abs(deltasOut.getByIndex(d, h, w).doubleValue())), mathContext20); // наращиваем ошибку

                            // если ответ сети совпадает с идеальным - прибавляем 1 к статистике
                            if (Math.round(trueOutput.getByIndex(d, h, w).doubleValue()) == Math.round(networkOutput.getByIndex(d, h, w).doubleValue()))
                                statistic += 1;
                        }
                    }
                }



//                this.epsilon = this.epsilon.divide(BigDecimal.valueOf(1.2), mathContext20);



                this.backward(deltasOut, input); // обратное распространение

                this.updateWeights(); // обновление весов
            }



//            this.epsilon = this.epsilon.divide(BigDecimal.valueOf(1.2), mathContext20);
//            this.epsilon = this.epsilon.divide(BigDecimal.valueOf(Math.sqrt(n + 1.)), mathContext20);



            System.out.println();
            System.out.println(n + " Error: " + error); // выводим значение ошибки
            System.out.println(n + " Statistic: " + (double) statistic / (this.outputSize.depth*this.outputSize.height*this.outputSize.width * inputPaths.size())); // выводим значение статистики
            System.out.println();
            // если все ответы сети совпадают с идеальными, то обучение останавливаеться
            if (statistic == this.outputSize.depth*this.outputSize.height*this.outputSize.width * inputPaths.size()) {
                System.out.println("!!! CONGRATULATIONS !!!");
                break;
            }
        }
    }

    public void testing(final ArrayList<String> inputPaths, final ArrayList<Integer> classNumbers) {
        Tensor input; // тензор для хранения входных изображений
        Tensor trueOutput = new Tensor(this.outputSize); // тензор для хранения идеальных выходов
        Tensor networkOutput; // тензор выходов нейронной сети
        Tensor deltasOut = new Tensor(this.outputSize); // тензор выходных дельт
        BigDecimal error = BigDecimal.ZERO;
        int statistic = 0;
        // проходимся по входным тензорам
        for (int i = 0; i < inputPaths.size(); i++) {
            input = this.getTensorFromImage(inputPaths.get(i), this.inputSize); // инициализируем входной тензор
            // инициализируем идеальный выходной тензор
            for (int j = 1; j <= this.outputSize.depth*this.outputSize.height*this.outputSize.width; j++) {
                if (j == classNumbers.get(i))
                    trueOutput.setByIndex(0, 0, j - 1, BigDecimal.ONE);
                else
                    trueOutput.setByIndex(0, 0, j - 1, BigDecimal.ZERO);
            }

            networkOutput = this.forward(input); // прямое распространение + получаем выход сети

            // проходимся по выходному тензору
            for (int d = 0; d < this.outputSize.depth; d++) {
                for (int h = 0; h < this.outputSize.height; h++) {
                    for (int w = 0; w < this.outputSize.width; w++) {
                        if (trueOutput.getByIndex(d, h, w).compareTo(BigDecimal.ONE) == 0)
                            networkOutput.setByIndex(d, h, w, BigDecimal.ONE.subtract(networkOutput.getByIndex(d, h, w), mathContext20));

                        // считаем дельты по выходу
                        deltasOut.setByIndex(d, h, w, trueOutput.getByIndex(d, h, w)
                                .subtract(networkOutput.getByIndex(d, h, w), mathContext20));



                        System.out.println(deltasOut.getByIndex(d, h, w));



                        error = error.add(BigDecimal.valueOf(Math.abs(deltasOut.getByIndex(d, h, w).doubleValue())), mathContext20); // наращиваем ошибку

                        // если ответ сети совпадает с идеальным - прибавляем 1 к статистике
                        if (Math.round(trueOutput.getByIndex(d, h, w).doubleValue()) == Math.round(networkOutput.getByIndex(d, h, w).doubleValue()))
                            statistic += 1;
                    }
                }
            }
        }

        System.out.println();
        System.out.println("testing Error: " + error); // выводим значение ошибки
        System.out.println("testing Statistic: " + (double) statistic / (this.outputSize.depth*this.outputSize.height*this.outputSize.width * inputPaths.size())); // выводим значение статистики
        System.out.println();
    }

    /*public Tensor forward(final Tensor input) {
        // прямое распространение по всем слоям
        this.forwardOutputs[0] = this.convolutions[0].forward(input);
        this.forwardOutputs[1] = this.convolutions[1].forward(this.forwardOutputs[0]);

        this.forwardOutputs[2] = this.maxPoolings[0].forward(this.forwardOutputs[1]);

        this.forwardOutputs[3] = this.convolutions[2].forward(this.forwardOutputs[2]);
        this.forwardOutputs[4] = this.convolutions[3].forward(this.forwardOutputs[3]);

        this.forwardOutputs[5] = this.maxPoolings[1].forward(this.forwardOutputs[4]);

        this.forwardOutputs[6] = this.convolutions[4].forward(this.forwardOutputs[5]);
        this.forwardOutputs[7] = this.convolutions[5].forward(this.forwardOutputs[6]);
        this.forwardOutputs[8] = this.convolutions[6].forward(this.forwardOutputs[7]);

        this.forwardOutputs[9] = this.maxPoolings[2].forward(this.forwardOutputs[8]);

        this.forwardOutputs[10] = this.convolutions[7].forward(this.forwardOutputs[9]);
        this.forwardOutputs[11] = this.convolutions[8].forward(this.forwardOutputs[10]);
        this.forwardOutputs[12] = this.convolutions[9].forward(this.forwardOutputs[11]);

        this.forwardOutputs[13] = this.maxPoolings[3].forward(this.forwardOutputs[12]);

        this.forwardOutputs[14] = this.convolutions[10].forward(this.forwardOutputs[13]);
        this.forwardOutputs[15] = this.convolutions[11].forward(this.forwardOutputs[14]);
        this.forwardOutputs[16] = this.convolutions[12].forward(this.forwardOutputs[15]);

        this.forwardOutputs[17] = this.maxPoolings[4].forward(this.forwardOutputs[16]);

        this.forwardOutputs[18] = this.fullyNecteds[0].forward(this.forwardOutputs[17]);
        this.forwardOutputs[19] = this.fullyNecteds[1].forward(this.forwardOutputs[18]);
        this.forwardOutputs[20] = this.fullyNecteds[2].forward(this.forwardOutputs[19]);

//        return this.softmax(this.forwardOutputs[20]); // применяем softmax и возвращаем выходной тензор
        return this.forwardOutputs[20];
    }*/
    public Tensor forward(final Tensor input) {
        // прямое распространение по всем слоям
        this.forwardOutputs[0] = this.convolutions[0].forward(input);

        this.forwardOutputs[1] = this.maxPoolings[0].forward(this.forwardOutputs[0]);

        this.forwardOutputs[2] = this.convolutions[1].forward(this.forwardOutputs[1]);

        this.forwardOutputs[3] = this.maxPoolings[1].forward(this.forwardOutputs[2]);

        this.forwardOutputs[4] = this.convolutions[2].forward(this.forwardOutputs[3]);
        this.forwardOutputs[5] = this.convolutions[3].forward(this.forwardOutputs[4]);

        this.forwardOutputs[6] = this.maxPoolings[2].forward(this.forwardOutputs[5]);

        this.forwardOutputs[7] = this.convolutions[4].forward(this.forwardOutputs[6]);
        this.forwardOutputs[8] = this.convolutions[5].forward(this.forwardOutputs[7]);

        this.forwardOutputs[9] = this.maxPoolings[3].forward(this.forwardOutputs[8]);

        this.forwardOutputs[10] = this.convolutions[6].forward(this.forwardOutputs[9]);
        this.forwardOutputs[11] = this.convolutions[7].forward(this.forwardOutputs[10]);

        this.forwardOutputs[12] = this.maxPoolings[4].forward(this.forwardOutputs[11]);

        this.forwardOutputs[13] = this.fullyNecteds[0].forward(this.forwardOutputs[12]);
        this.forwardOutputs[14] = this.fullyNecteds[1].forward(this.forwardOutputs[13]);
        this.forwardOutputs[15] = this.fullyNecteds[2].forward(this.forwardOutputs[14], true);

        return this.forwardOutputs[15];
    }

    /*public void backward(final Tensor deltasOut, final Tensor input) {
        Tensor dOut; // тензор для временного хранения дельт для предыдущих слоев

        // обратное распространение по всем слоям
        dOut = this.fullyNecteds[2].backward(deltasOut, this.forwardOutputs[19]);
        dOut = this.fullyNecteds[1].backward(dOut, this.forwardOutputs[18]);
        dOut = this.fullyNecteds[0].backward(dOut, this.forwardOutputs[17]);

        dOut = this.maxPoolings[4].backward(dOut);

        dOut = this.convolutions[12].backward(dOut, this.forwardOutputs[15]);
        dOut = this.convolutions[11].backward(dOut, this.forwardOutputs[14]);
        dOut = this.convolutions[10].backward(dOut, this.forwardOutputs[13]);

        dOut = this.maxPoolings[3].backward(dOut);

        dOut = this.convolutions[9].backward(dOut, this.forwardOutputs[11]);
        dOut = this.convolutions[8].backward(dOut, this.forwardOutputs[10]);
        dOut = this.convolutions[7].backward(dOut, this.forwardOutputs[9]);

        dOut = this.maxPoolings[2].backward(dOut);

        dOut = this.convolutions[6].backward(dOut, this.forwardOutputs[7]);
        dOut = this.convolutions[5].backward(dOut, this.forwardOutputs[6]);
        dOut = this.convolutions[4].backward(dOut, this.forwardOutputs[5]);

        dOut = this.maxPoolings[1].backward(dOut);

        dOut = this.convolutions[3].backward(dOut, this.forwardOutputs[3]);
        dOut = this.convolutions[2].backward(dOut, this.forwardOutputs[2]);

        dOut = this.maxPoolings[0].backward(dOut);

        dOut = this.convolutions[1].backward(dOut, this.forwardOutputs[0]);
        this.convolutions[0].backward(dOut, input, true);
    }*/
    public void backward(final Tensor deltasOut, final Tensor input) {
        Tensor dOut; // тензор для временного хранения дельт для предыдущих слоев

        // обратное распространение по всем слоям
        dOut = this.fullyNecteds[2].backward(deltasOut, this.forwardOutputs[14], this.forwardOutputs[15]);
        dOut = this.fullyNecteds[1].backward(dOut, this.forwardOutputs[13]);
        dOut = this.fullyNecteds[0].backward(dOut, this.forwardOutputs[12]);

        dOut = this.maxPoolings[4].backward(dOut);

        dOut = this.convolutions[7].backward(dOut, this.forwardOutputs[10]);
        dOut = this.convolutions[6].backward(dOut, this.forwardOutputs[9]);

        dOut = this.maxPoolings[3].backward(dOut);

        dOut = this.convolutions[5].backward(dOut, this.forwardOutputs[7]);
        dOut = this.convolutions[4].backward(dOut, this.forwardOutputs[6]);

        dOut = this.maxPoolings[2].backward(dOut);

        dOut = this.convolutions[3].backward(dOut, this.forwardOutputs[4]);
        dOut = this.convolutions[2].backward(dOut, this.forwardOutputs[3]);

        dOut = this.maxPoolings[1].backward(dOut);

        dOut = this.convolutions[1].backward(dOut, this.forwardOutputs[1]);

        dOut = this.maxPoolings[0].backward(dOut);

        this.convolutions[0].backward(dOut, input, true);
    }

    public void updateWeights() {
        // обновляем веса по всем слоям
        for (Convolution convolution : this.convolutions)
            convolution.updateWeights(this.epsilon, this.alpha);
        for (FullyNected fullyNected : this.fullyNecteds)
            fullyNected.updateWeights(this.epsilon, this.alpha);
    }

    public Tensor softmax(final Tensor input) {
        Tensor output = new Tensor(input.getSize()); // создаем выходной тензор

        BigDecimal sum = BigDecimal.ZERO;
        // проходимся по входному тензору
        for (int d = 0; d < input.getDepth(); d++) {
            for (int h = 0; h < input.getHeight(); h++) {
                for (int w = 0; w < input.getWidth(); w++) {
                    output.setByIndex(d, h, w, BigDecimal.valueOf(Math.exp(input.getByIndex(d, h, w).doubleValue()))); // записываем в тензор экспоненты в степени
                    sum = sum.add(output.getByIndex(d, h, w), mathContext20); // наращиваем сумму
                }
            }
        }
        // проходимся по входному тензору
        for (int d = 0; d < input.getDepth(); d++) {
            for (int h = 0; h < input.getHeight(); h++) {
                for (int w = 0; w < input.getWidth(); w++)
                    output.setByIndex(d, h, w, output.getByIndex(d, h, w).divide(sum, mathContext20)); // расчитываем и записываем выходные значения
            }
        }

        return output; // возвращаем выходной тензор
    }

    public Tensor derivativeSoftmax(final Tensor input) {
        Tensor deltasOut = new Tensor(input.getSize()); // создаем выходной вектор дельт
        // проходимся по выходному тензору
        for (int d = 0; d < this.outputSize.depth; d++) {
            for (int h = 0; h < this.outputSize.depth; h++) {
                for (int w = 0; w < this.outputSize.depth; w++) {
                    deltasOut.setByIndex(d, h, w, input.getByIndex(d, h, w)
                            .multiply(BigDecimal.ONE.subtract(input.getByIndex(d, h, w), mathContext20), mathContext20));
                }
            }
        }
        return deltasOut; // возвращаем дельты
    }

    public Tensor getTensorFromImage(String path, TensorSize size) {

        Tensor output = new Tensor(size); // создаем выходной тензор

        BufferedImage bufferedImage;
        try {
            bufferedImage = ImageIO.read(new File(path)); // записываем изображение

            // изменяем размер, если нужно
            if (bufferedImage.getWidth() != size.width || bufferedImage.getHeight() != size.height)
                bufferedImage = this.resize(bufferedImage, size.width, size.height);

            // инициализируем выходные тензор
            for (int h = 0; h < size.height; h++) {
                for (int w = 0; w < size.width; w++) {
                    output.setByIndex(0, h, w, BigDecimal.valueOf(((bufferedImage.getRGB(h, w) >> 16) & 255) / 255.)); // Red
                    output.setByIndex(1, h, w, BigDecimal.valueOf(((bufferedImage.getRGB(h, w) >> 8) & 255) / 255.)); // Green
                    output.setByIndex(2, h, w, BigDecimal.valueOf((bufferedImage.getRGB(h, w) & 255) / 255.)); // Blue
                }
            }

            // очистка памяти
            bufferedImage.flush();
            bufferedImage = null;

            return output; // возвращаем изображение в тензоре
        }
        catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }

    public BufferedImage resize(BufferedImage img, int newW, int newH) {
        Image tmp = img.getScaledInstance(newW, newH, Image.SCALE_SMOOTH);
        BufferedImage resizedImage = new BufferedImage(newW, newH, BufferedImage.TYPE_INT_ARGB);

        Graphics2D g2d = resizedImage.createGraphics();
        g2d.drawImage(tmp, 0, 0, null);
        g2d.dispose();

        return resizedImage;
    }
}
