import static org.bytedeco.javacpp.opencv_highgui.WINDOW_AUTOSIZE;
import static org.bytedeco.javacpp.opencv_highgui.imshow;
import static org.bytedeco.javacpp.opencv_highgui.namedWindow;
import static org.bytedeco.javacpp.opencv_highgui.waitKey;
import static org.bytedeco.javacpp.opencv_imgcodecs.imread;

import javax.swing.JFrame;

import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.javacv.OpenCVFrameConverter.ToMat;

public class LoadAndDisplay {

	public static void Show(Mat mat, String title) {
		//
		ToMat converter = new OpenCVFrameConverter.ToMat();
		CanvasFrame canvas = new CanvasFrame(title, 1);
		canvas.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		canvas.showImage(converter.convert(mat));

	}

	public static void main(String[] args) {
		//
		Mat image = imread("data/tower.jpg", 1);
		if (image == null || image.empty()) {
			System.out.println("fail");
			return;
		}

		System.out.println("image" + image.cols() + "	x	" + image.rows());
		/**	
			*/
		Show(image, "img");
		waitKey(0); // Wait for a keystroke in the window

	}

}