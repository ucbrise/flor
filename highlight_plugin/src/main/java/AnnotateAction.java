import com.intellij.lang.Language;
import com.intellij.openapi.actionSystem.AnAction;
import com.intellij.openapi.actionSystem.AnActionEvent;
import com.intellij.openapi.actionSystem.CommonDataKeys;
import com.intellij.openapi.actionSystem.DataContext;
import com.intellij.openapi.command.WriteCommandAction;
import com.intellij.openapi.editor.*;
import com.intellij.openapi.editor.ex.EditorEx;
import com.intellij.openapi.editor.highlighter.EditorHighlighterFactory;
import com.intellij.openapi.fileEditor.FileDocumentManager;
import com.intellij.openapi.fileTypes.FileType;
import com.intellij.openapi.project.Project;
import com.intellij.openapi.ui.DialogBuilder;
import com.intellij.openapi.ui.DialogWrapper;
import com.intellij.openapi.vfs.VirtualFile;
import com.intellij.psi.PsiElement;
import com.intellij.psi.PsiFile;

import javax.swing.*;
import java.awt.*;

public class AnnotateAction extends AnAction {
    @Override
    public void actionPerformed(AnActionEvent e) {
        Project project = e.getProject();
        DataContext dataContext = e.getDataContext();
        Editor editor = e.getData(CommonDataKeys.EDITOR);
        Document document = editor.getDocument();
        // Create read-only guarded block for current document
        // document.createGuardedBlock(0, document.getLineEndOffset(document.getLineCount() - 1));
        PsiFile file = e.getData(CommonDataKeys.PSI_FILE);
        Language lang = file.getLanguage();

        // Open the raw script that implements the definition of the selection
        CaretModel caretModel = editor.getCaretModel();
        SelectionModel selectionModel = editor.getSelectionModel();

        Caret currentCaret = caretModel.getCurrentCaret();
        boolean hasSelection = currentCaret.hasSelection();

        String selectedText = currentCaret.getSelectedText();

        int caretOffset = caretModel.getOffset();

        PsiElement element = file.findElementAt(caretOffset);
        String elementText = selectedText;
        if (!hasSelection && element != null) {
            elementText = element.getText();
        }

        /*
        PyGotoDeclarationHandler handler = new PyGotoDeclarationHandler();
        PsiElement elementReference = handler.getGotoDeclarationTarget(element, editor);
        PsiReference elementReference = element.getReference();
        PsiMethod containingMethod = PsiTreeUtil.getParentOfType(element, PsiMethod.class);

        // Open Source File if the selection has one
        if (elementReference != null) {
            VirtualFile containingFile = elementReference.getContainingFile().getVirtualFile();
            FileEditorManager.getInstance(project).openFile(containingFile, true);
        }
         */

        String label = "";


        // Create read-only editor view
        FileType fileType = lang.getAssociatedFileType();
        final EditorFactory editorFactory = EditorFactory.getInstance();
        Document labelDocument = editorFactory.createDocument("");
        Editor labelEditor = editorFactory.createEditor(labelDocument);

        JTextField textField = new JTextField();

        try {
            if (fileType != null) {
                ((EditorEx) labelEditor).setHighlighter(EditorHighlighterFactory.getInstance().createEditorHighlighter(project, fileType));
            }

            // JComponent component = labelEditor.getComponent();
            textField.setPreferredSize(new Dimension(640, 28));

            DialogBuilder db = new DialogBuilder(project);

            db.title("Label for " + elementText).dimensionKey("GuiDesigner.FormSource.Dialog");
            db.centerPanel(textField).setPreferredFocusComponent(textField);

            db.addOkAction();
            if (db.show() == DialogWrapper.OK_EXIT_CODE) {
                // Extract label name
                // label = labelDocument.getText();
                label = textField.getText();
            }
        } finally {
            editorFactory.releaseEditor(labelEditor);
        }

        if (label.length() > 0) {
            String replaceText = "GET(" + "\"" + label + "\"" + ", " + elementText + ")";
            Document temp = EditorFactory.getInstance().createDocument(replaceText);
            // PsiFile tempPsiFile = PsiDocumentManager.getInstance(project).getPsiFile(temp);

            // Modify PSI element
            WriteCommandAction.runWriteCommandAction(project, () -> {
                document.setReadOnly(false);
                if (hasSelection) {
                    // Replace the text at caret
                    int start = selectionModel.getSelectionStart();
                    int end = selectionModel.getSelectionEnd();
                    document.replaceString(start, end, replaceText);
                    selectionModel.removeSelection();
                } else {
                    /*
                    PsiFile tempPsiFile = PsiFileFactory.getInstance(project).createFileFromText(element.getLanguage(), replaceText);
                    PsiElement tempElement = tempPsiFile.findElementAt(0);
                    PsiElement parentElement = null;
                    if (tempElement != null) {
                        parentElement = tempElement.getParent();
                    }
                    while (parentElement != null) {
                        tempElement = parentElement;
                        parentElement = tempElement.getParent();
                    }
                    if (tempElement != null) {
                        element.replace(tempElement);
                    }
                     */
                    int start = element.getTextOffset();
                    int end = start + element.getTextLength();
                    document.replaceString(start, end, replaceText);

                    // DONE: Auto save document
                    FileDocumentManager.getInstance().saveDocument(document);

                }
                // document.createGuardedBlock(0, document.getTextLength());
            });
        }
    }

    @Override
    public void update(AnActionEvent e) {
        Editor editor = e.getRequiredData(CommonDataKeys.EDITOR);

//        //        Only enable when hasSelection
//        CaretModel caretModel = editor.getCaretModel();
//        e.getPresentation().setEnabledAndVisible(caretModel.getCurrentCaret().hasSelection());

        PsiFile file = e.getData(CommonDataKeys.PSI_FILE);
        VirtualFile vFile;
        if (file != null) {
            vFile = file.getVirtualFile();
            boolean inHighlightedFolder = vFile.getPath().contains("highlighted.");
            e.getPresentation().setEnabledAndVisible(inHighlightedFolder);
            if (inHighlightedFolder) {
                ToggleReadOnlyView.toggleReadOnlyFor(editor.getDocument());
            }
        }
    }
}
