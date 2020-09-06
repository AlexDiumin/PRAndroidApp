import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.Objects;

public class Main {

    private static final TensorSize inputSize = new TensorSize(3, 224, 224); // длинна и ширина входных изображений
    private static final int classesCount = 2; // кол-во классов

    public static void main(String[] args) {
        ArrayList<Tensor> input = new ArrayList<>();
        ArrayList<Tensor> output = new ArrayList<>();

        int inputCount;
        ArrayList<BufferedImage> bufferedImages = new ArrayList<>();
        for (int f = 0; f <= classesCount; f++) {
            listFilesForFolder(new File(String.valueOf(f)), bufferedImages); // считываем с папки в массив

            inputCount = input.size();
            // данные с массива переносим во входной и выходной тензоры
            for (int i = inputCount; i < inputCount + bufferedImages.size(); i++) {
                input.add(new Tensor(inputSize));
                output.add(new Tensor(1, 1, classesCount));

                // инициализируем входные тензоры
                for (int h = 0; h < inputSize.height; h++) {
                    for (int w = 0; w < inputSize.width; w++) {
                        input.get(i).setByIndex(0, h, w, BigDecimal.valueOf(new Color(bufferedImages.get(i - inputCount).getRGB(h, w)).getRed()/255.));
                        input.get(i).setByIndex(1, h, w, BigDecimal.valueOf(new Color(bufferedImages.get(i - inputCount).getRGB(h, w)).getGreen()/255.));
                        input.get(i).setByIndex(2, h, w, BigDecimal.valueOf(new Color(bufferedImages.get(i - inputCount).getRGB(h, w)).getBlue()/255.));
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

            bufferedImages.clear(); // очищаем массив изображений
        }

        bufferedImages = null; // очистка памяти

        Tensor[] in = input.toArray(Tensor[]::new); // преобразование в массив
        input.clear(); // очистка памяти
        input = null; // очистка памяти
        Tensor[] out = output.toArray(Tensor[]::new); // преобразование в массив
        output.clear(); // очистка памяти
        output = null; // очистка памяти

        new NeuralNetwork(inputSize, classesCount).training(in, out); // создаем нейросеть и начинаем обучение
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
}
