import com.intellij.execution.ExecutionException;
import com.intellij.execution.configurations.GeneralCommandLine;
import com.intellij.execution.process.OSProcessHandler;
import com.intellij.execution.process.ProcessHandler;

import java.nio.charset.Charset;
import java.util.List;

public class RunCommandLine {
    public static ProcessHandler getProcessHandler(List<String> command, String workDir) {
        ProcessHandler processHandler = null;
        GeneralCommandLine generalCommandLine = new GeneralCommandLine(command);
        generalCommandLine.setCharset(Charset.forName("UTF-8"));
        generalCommandLine.setWorkDirectory(workDir);

        try {
            processHandler = new OSProcessHandler(generalCommandLine);
            processHandler.startNotify();
        } catch (ExecutionException ex) {
            ex.printStackTrace();
        }

        return processHandler;
    }
}
