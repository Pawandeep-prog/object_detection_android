package com.programminghut.object_detection

import android.content.Intent
import android.graphics.*
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.os.FileUtils
import android.provider.MediaStore
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import com.programminghut.object_detection.ml.SsdMobilenetV11Metadata1
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp

class MainActivity : AppCompatActivity() {

    val paint = Paint()
    lateinit var btn: Button
    lateinit var imageView: ImageView
    lateinit var bitmap: Bitmap
    var colors = listOf<Int>(Color.BLUE, Color.GREEN, Color.RED, Color.CYAN, Color.GRAY, Color.BLACK,
                                        Color.DKGRAY, Color.MAGENTA, Color.YELLOW, Color.RED)
    lateinit var labels: List<String>
    lateinit var model: SsdMobilenetV11Metadata1
    val imageProcessor = ImageProcessor.Builder().add(ResizeOp(300, 300, ResizeOp.ResizeMethod.BILINEAR)).build()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        labels = FileUtil.loadLabels(this, "labels.txt")
        model = SsdMobilenetV11Metadata1.newInstance(this)

        paint.setColor(Color.BLUE)
        paint.style = Paint.Style.STROKE
        paint.strokeWidth = 5.0f
//        paint.textSize = paint.textSize*3

        Log.d("labels", labels.toString())

        val intent = Intent()
        intent.setAction(Intent.ACTION_GET_CONTENT)
        intent.setType("image/*")

        btn = findViewById(R.id.btn)
        imageView = findViewById(R.id.imaegView)

        btn.setOnClickListener {
            startActivityForResult(intent, 101)
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if(requestCode == 101){
            var uri = data?.data
            bitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, uri)
            get_predictions()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        model.close()
    }

    fun get_predictions(){

        var image = TensorImage.fromBitmap(bitmap)
        image = imageProcessor.process(image)
        val outputs = model.process(image)
        val locations = outputs.locationsAsTensorBuffer.floatArray
        val classes = outputs.classesAsTensorBuffer.floatArray
        val scores = outputs.scoresAsTensorBuffer.floatArray
        val numberOfDetections = outputs.numberOfDetectionsAsTensorBuffer.floatArray



        val mutable = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        var canvas = Canvas(mutable)
        var h = mutable.height
        var w = mutable.width


        paint.textSize = h/15f
        paint.strokeWidth = h/85f
        scores.forEachIndexed { index, fl ->
            if(fl > 0.5){
                var x = index
                x *= 4
                paint.setColor(colors.get(index))
                paint.style = Paint.Style.STROKE
                canvas.drawRect(RectF(locations.get(x+1)*w, locations.get(x)*h, locations.get(x+3)*w, locations.get(x+2)*h), paint)
                paint.style = Paint.Style.FILL
                canvas.drawText(labels[classes.get(index).toInt()] + " " + fl.toString(), locations.get(x+1)*w, locations.get(x)*h, paint)
            }
        }

        imageView.setImageBitmap(mutable)

    }
}