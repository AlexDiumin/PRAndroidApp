import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.Objects;

public class Main {

    private static final TensorSize inputSize = new TensorSize(3, 160, 160); // длинна и ширина входных изображений
    private static final int classesCount = 2; // кол-во классов

    public static void main(String[] args) {
        ArrayList<String> inputPaths = new ArrayList<>(); // массив путей к изображеням
        ArrayList<Integer> classNumbers = new ArrayList<>(); // массив номеров классов

        arraysFromFolder(inputPaths, classNumbers, new File("training"));

        NeuralNetwork convNet = new NeuralNetwork(inputSize, classesCount); // создаем нейросеть
        convNet.training(inputPaths, classNumbers); // начинаем обучение

        // очистка памяти
        inputPaths.clear();
        inputPaths = null;
        classNumbers.clear();
        classNumbers = null;

        System.out.println("\n ===== Testing =====");

        inputPaths = new ArrayList<>();
        classNumbers = new ArrayList<>();

        arraysFromFolder(inputPaths, classNumbers, new File("testing"));

        convNet.testing(inputPaths, classNumbers); // тестирование
    }

    public static void readFromFolder(ArrayList<Tensor> input, ArrayList<Tensor> output, String folderName) {
        int inputCount;
        ArrayList<BufferedImage> bufferedImages = new ArrayList<>();
        for (int f = 0; f <= classesCount; f++) {
            listFilesForFolder(new File(folderName + "/" + f), bufferedImages); // считываем с папки в массив
            inputCount = input.size();
            // данные с массива переносим во входной и выходной тензоры
            for (int i = inputCount; i < inputCount + bufferedImages.size(); i++) {
                input.add(new Tensor(inputSize));
                output.add(new Tensor(1, 1, classesCount));

                // инициализируем входные тензоры
                for (int h = 0; h < inputSize.height; h++) {
                    for (int w = 0; w < inputSize.width; w++) {
                        input.get(i).setByIndex(0, h, w, BigDecimal.valueOf(((bufferedImages.get(i - inputCount).getRGB(h, w) >> 16) & 255) / 255.)); // Red
                        input.get(i).setByIndex(1, h, w, BigDecimal.valueOf(((bufferedImages.get(i - inputCount).getRGB(h, w) >> 8) & 255) / 255.)); // Green
                        input.get(i).setByIndex(2, h, w, BigDecimal.valueOf((bufferedImages.get(i - inputCount).getRGB(h, w) & 255) / 255.)); // Blue
                    }
                }

                // инициализируем выходные тензоры
                for (int j = 1; j <= classesCount; j++) {
                    if (j == f)
                        output.get(i).setByIndex(0, 0, j - 1, BigDecimal.ONE);
                    else
                        output.get(i).setByIndex(0, 0, j - 1, BigDecimal.ZERO);
                }
            }

            // очищаем массив изображений
            for (int i = bufferedImages.size() - 1; i >= 0; i--)
                bufferedImages.get(i).flush();
            bufferedImages.clear();


            System.out.println(folderName + "/" + f + " - READY\n");
        }
        bufferedImages = null; // очистка памяти
    }

    public static void listFilesForFolder(final File folder, ArrayList<BufferedImage> bufferedImages) {
        for (final File fileEntry : Objects.requireNonNull(folder.listFiles())) {
            if (fileEntry.isDirectory())
                listFilesForFolder(fileEntry, bufferedImages);
            else {
                try {
                    BufferedImage bufferedImage = ImageIO.read(fileEntry);
                    if ( !(bufferedImage.getWidth() == inputSize.width && bufferedImage.getHeight() == inputSize.height) )
                        bufferedImage = resize(bufferedImage, inputSize.width, inputSize.height);

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

    public static void arraysFromFolder(ArrayList<String> inputPaths, ArrayList<Integer> classNumbers, final File folder) {
        for (final File fileEntry : Objects.requireNonNull(folder.listFiles())) {
            if (fileEntry.isDirectory())
                arraysFromFolder(inputPaths, classNumbers, fileEntry);
            else {
                inputPaths.add(fileEntry.getPath()); // сохраняем путь к изображению

                for (int c = 0; c <= classesCount; c++) {
                    if (inputPaths.get(inputPaths.size() - 1).contains("/" + c + "/")
                            || inputPaths.get(inputPaths.size() - 1).contains("\\" + c + "\\")) {
                        classNumbers.add(c); // сохраняем номер класса изображения
                        break;
                    }
                }
            }
        }
    }
}
