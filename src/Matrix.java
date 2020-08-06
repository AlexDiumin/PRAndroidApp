import java.math.BigDecimal;
import java.util.Vector;

public class Matrix {

    private int rows; // число строк
    private int columns; // число столбцов
    private Vector<Vector<BigDecimal>> values; // значения

    // конструктор из заданных размеров
    public Matrix(int rows, int columns) {
        this.rows = rows; // сохраняем число строк
        this.columns = columns; // сохраняем число столбцов

        // создаем вектор для значений матрицы
        values = new Vector<>();
        for (int r = 0; r < rows; r++) {
            values.add(new Vector<>());
            for (int c = 0; c < columns; c++)
                values.get(r).add(BigDecimal.ZERO);
        }
    }

    public final BigDecimal getByIndex(int r, int c) {
        return values.get(r).get(c);
    }

    public void setByIndex(int r, int c, BigDecimal value) {
        values.get(r).set(c, value);
    }
}
