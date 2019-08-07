import com.intellij.ide.BrowserUtil;
import com.intellij.lang.Language;
import com.intellij.openapi.actionSystem.*;
import com.intellij.openapi.editor.CaretModel;
import com.intellij.openapi.editor.Editor;
import com.intellij.psi.PsiFile;

public class SearchStackOverflow extends AnAction {

    @Override
    public void actionPerformed(AnActionEvent e) {
        PsiFile file = e.getData(CommonDataKeys.PSI_FILE);
        Language lang = null;
        if (file != null) {
            lang = file.getLanguage();
        }
        String languageTag = null;
        if (lang != null) {
            languageTag = "+[" + lang.getDisplayName().toLowerCase() + "]";
        }

        Editor editor = e.getRequiredData(CommonDataKeys.EDITOR);
        CaretModel caretModel = editor.getCaretModel();
        String selectedText = caretModel.getCurrentCaret().getSelectedText();

        String query = null;
        if (selectedText != null) {
            query = selectedText.replace(' ', '+') + languageTag;
        }
        BrowserUtil.browse("https://stackoverflow.com/search?q=" + query);
    }

    @Override
    public void update(AnActionEvent e) {
        Editor editor = e.getRequiredData(CommonDataKeys.EDITOR);
        CaretModel caretModel = editor.getCaretModel();
//        Only enable when hasSelection
        e.getPresentation().setEnabledAndVisible(caretModel.getCurrentCaret().hasSelection());
    }
}
