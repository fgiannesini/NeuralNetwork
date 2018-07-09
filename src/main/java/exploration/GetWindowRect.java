package exploration;

import com.sun.jna.platform.win32.User32;
import com.sun.jna.platform.win32.WinDef;

import java.awt.*;
import java.io.File;
import java.io.IOException;


public class GetWindowRect {

    public Rectangle getWindowRect() throws IOException {
        Desktop desktop = Desktop.getDesktop();
        desktop.open(new File("C:\\Program Files\\VideoLAN\\VLC\\vlc.exe"));
//            WinDef.HWND hwnd = User32.INSTANCE.FindWindow
//                    (null, "Lecteur Multimédia VLC");// window title
        WinDef.HWND hwnd = User32.INSTANCE.GetForegroundWindow();
        WinDef.RECT rect = new WinDef.RECT();
        if (hwnd == null) {
            System.out.println("Vlc is not running");
        } else {
//                User32.INSTANCE.ShowWindow(hwnd, 9);        // SW_RESTORE
//                User32.INSTANCE.SetForegroundWindow(hwnd);   // bring to front
            User32.INSTANCE.GetWindowRect(hwnd, rect);
            System.out.println("rect = " + rect);
        }
        return rect.toRectangle();
    }

    public static void main(String[] args) throws IOException {
        new GetWindowRect().getWindowRect();
    }
}
