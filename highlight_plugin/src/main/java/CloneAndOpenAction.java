import com.intellij.execution.ExecutionException;
import com.intellij.execution.configurations.GeneralCommandLine;
import com.intellij.execution.process.OSProcessHandler;
import com.intellij.execution.process.ProcessHandler;
import com.intellij.openapi.actionSystem.AnAction;
import com.intellij.openapi.actionSystem.AnActionEvent;
import com.intellij.openapi.actionSystem.CommonDataKeys;
import com.intellij.openapi.editor.Caret;
import com.intellij.openapi.editor.CaretModel;
import com.intellij.openapi.editor.Editor;
import com.intellij.openapi.editor.ScrollType;
import com.intellij.openapi.fileEditor.FileEditorManager;
import com.intellij.openapi.project.Project;
import com.intellij.openapi.project.ProjectManager;
import com.intellij.openapi.vfs.LocalFileSystem;
import com.intellij.openapi.vfs.VirtualFile;
import com.intellij.psi.PsiElement;
import com.intellij.psi.PsiFile;
import com.intellij.psi.PsiManager;
//import com.jetbrains.python.psi.impl.PyGotoDeclarationHandler;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.SystemIndependent;
//import org.python.core.PyInteger;
//import org.python.core.PyObject;
//import org.python.util.PythonInterpreter;

import java.io.*;
import java.nio.charset.Charset;
import java.util.ArrayList;

public class CloneAndOpenAction extends AnAction {

    @Override
    public void actionPerformed(@NotNull AnActionEvent e) {
        Project project = e.getProject();
        Editor editor = e.getData(CommonDataKeys.EDITOR);
        CaretModel caretModel = null;
        if (editor != null) {
            caretModel = editor.getCaretModel();
        }
        Caret currentCaret = null;
        if (caretModel != null) {
            currentCaret = caretModel.getCurrentCaret();
        }
        int currentCaretOffset = 0;
        if (currentCaret != null) {
            currentCaretOffset = currentCaret.getOffset();
        }

        PsiFile file = e.getData(CommonDataKeys.PSI_FILE);
        assert file != null;
        VirtualFile srcVFile = file.getVirtualFile();
        String currFolderPath = srcVFile.getParent().getPath();
        String currFolderName = srcVFile.getParent().getName();
        String srcFilePath = srcVFile.getPath();
        File srcFile = new File(srcFilePath);
        @SystemIndependent String projectBasePath = project.getBasePath();

        SessionManager sessionManager = new SessionManager(projectBasePath);
        String dstFolderPath = sessionManager.getLatestSessionFolderPath(e);

        File dstDir = new File(dstFolderPath);
        dstDir.mkdir();

        String srcFileNameRoot = file.getName().split(".py")[0];

        //TODO: Construct canonical file names with folders taken into account
        String dstFilePath = dstFolderPath + "/" + srcFileNameRoot + "_h.py";

        assert srcFile.exists();

        // String program = "from argparse import Namespace\nfrom flor.commands.cp import exec_cp\nexec_cp(Namespace(src='" + srcFilePath + "', dst='" + dstFilePath + "'))";
        String program = "from argparse import Namespace; from flor.commands.cp import exec_cp; exec_cp(Namespace(src='" + srcFilePath + "', dst='" + dstFilePath + "'))";

        try {
            File tempFile = File.createTempFile("flor_plugin_temp", ".py");
            BufferedWriter out = new BufferedWriter(new FileWriter(tempFile));
            out.write(program);
            out.close();
            String[] command = { "/bin/bash", "-c", "\"python " + tempFile.getAbsolutePath() + "\""};
            Process p = Runtime.getRuntime().exec("python " +  tempFile.getAbsolutePath());
            p.waitFor();

            // ProcessBuilder pb = new ProcessBuilder("python", tempFile.getAbsolutePath());
            // Process p = pb.start();

            BufferedReader in = new BufferedReader(new InputStreamReader(p.getInputStream()));
            String ret = in.readLine();
            System.out.println("RetVal: " + ret);
        } catch (IOException | InterruptedException ex) {
            ex.printStackTrace();
        }


        File dstFile = new File(dstFilePath);

        ProcessHandler processHandler = null;

        // Attempt to generate another time if method 1 does not work
        if (!dstFile.exists()) {

            File tempFile = null;
            try {
                tempFile = File.createTempFile("flor_plugin_temp", ".py");
                BufferedWriter out = new BufferedWriter(new FileWriter(tempFile));
                out.write(program);
                out.close();
            } catch (IOException ex) {
                ex.printStackTrace();
            }

            // https://stackoverflow.com/questions/36853427/intellij-plugin-run-console-command
            ArrayList<String> command = new ArrayList<>();
            command.add("/bin/bash");
            command.add("-c");
            if (tempFile != null) {
                command.add("python " + tempFile.getAbsolutePath());
            }

            processHandler = RunCommandLine.getProcessHandler(command, projectBasePath);

        }

        // Must wait for the process to finish
        if (processHandler != null) {
            processHandler.waitFor();
        }

        assert dstFile.exists() : dstFilePath + " not found.";

        VirtualFile dstVFile = LocalFileSystem.getInstance().refreshAndFindFileByIoFile(dstFile);

        // Open Source File in new editor tab
        assert dstVFile != null : dstFilePath + " not found.";
        FileEditorManager.getInstance(project).openFile(dstVFile, true);
        Editor selectedTextEditor = FileEditorManager.getInstance(project).getSelectedTextEditor();
        if (selectedTextEditor != null) {
            selectedTextEditor.getCaretModel().moveToOffset((int) (currentCaretOffset + dstVFile.getLength() - srcVFile.getLength()));
            selectedTextEditor.getScrollingModel().scrollToCaret(ScrollType.CENTER);
            ToggleReadOnlyView.toggleReadOnlyFor(selectedTextEditor.getDocument());
        }



    }

    @Override
    public void update(@NotNull AnActionEvent e) {
        super.update(e);
        Editor editor = e.getData(CommonDataKeys.EDITOR);
        PsiFile file = e.getData(CommonDataKeys.PSI_FILE);
        VirtualFile vFile;
        if (file != null) {
            vFile = file.getVirtualFile();
            boolean inHighlightedFolder = vFile.getPath().contains("highlighted.");
            if (inHighlightedFolder) {
                if (editor != null) {
                    ToggleReadOnlyView.toggleReadOnlyFor(editor.getDocument());
                }
            }
        }
    }
}
