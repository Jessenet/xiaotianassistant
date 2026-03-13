package com.xiaotian.assistant

import android.content.Context
import android.content.Intent
import android.provider.Settings
import android.text.TextUtils

/**
 * 无障碍服务辅助类
 */
object AccessibilityHelper {
    
    /**
     * 检查无障碍服务是否已启用
     */
    fun isAccessibilityServiceEnabled(context: Context): Boolean {
        val service = "${context.packageName}/${QQMusicAccessibilityService::class.java.canonicalName}"
        
        try {
            val enabledServices = Settings.Secure.getString(
                context.contentResolver,
                Settings.Secure.ENABLED_ACCESSIBILITY_SERVICES
            )
            
            if (!TextUtils.isEmpty(enabledServices)) {
                val colonSplitter = TextUtils.SimpleStringSplitter(':')
                colonSplitter.setString(enabledServices)
                
                while (colonSplitter.hasNext()) {
                    val componentName = colonSplitter.next()
                    if (componentName.equals(service, ignoreCase = true)) {
                        return true
                    }
                }
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }
        
        return false
    }
    
    /**
     * 打开无障碍设置页面
     */
    fun openAccessibilitySettings(context: Context) {
        try {
            val intent = Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS)
            intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            context.startActivity(intent)
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }
    
    /**
     * 检查服务是否正在运行
     */
    fun isServiceRunning(): Boolean {
        return QQMusicAccessibilityService.isServiceEnabled()
    }
    
    /**
     * 获取服务实例
     */
    fun getServiceInstance(): QQMusicAccessibilityService? {
        return QQMusicAccessibilityService.getInstance()
    }
}
