package com.xiaotian.assistant

import android.app.Application
import android.util.Log

/**
 * Application类 - 用于全局初始化和错误捕获
 */
class BoltApplication : Application() {
    
    companion object {
        private const val TAG = "BoltApplication"
    }
    
    override fun onCreate() {
        super.onCreate()
        
        try {
            Log.d(TAG, "=== Bolt Application 启动 ===")
            Log.d(TAG, "Android版本: ${android.os.Build.VERSION.SDK_INT}")
            Log.d(TAG, "设备型号: ${android.os.Build.MODEL}")
            Log.d(TAG, "可用内存: ${Runtime.getRuntime().maxMemory() / 1024 / 1024}MB")
            
            // 保存默认异常处理器
            val defaultHandler = Thread.getDefaultUncaughtExceptionHandler()
            
            // 设置全局异常处理器
            Thread.setDefaultUncaughtExceptionHandler { thread, throwable ->
                // 语音识别线程在服务停止时的正常竞态异常，不应崩溃
                if (throwable is RuntimeException && throwable.message?.contains("error reading audio buffer") == true) {
                    Log.w(TAG, "捕获到音频读取异常（服务停止时的正常竞态），已忽略", throwable)
                    return@setDefaultUncaughtExceptionHandler
                }
                
                Log.e(TAG, "!!! 未捕获异常 !!!", throwable)
                Log.e(TAG, "线程: ${thread.name}")
                Log.e(TAG, "错误: ${throwable.javaClass.simpleName}: ${throwable.message}")
                throwable.printStackTrace()
                
                // 调用原始默认处理器
                defaultHandler?.uncaughtException(thread, throwable)
            }
            
            Log.d(TAG, "Application初始化完成")
            
        } catch (e: Exception) {
            Log.e(TAG, "Application初始化失败", e)
        }
    }
}
